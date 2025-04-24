import math
from typing import List, Optional, NamedTuple

import numpy as np

import torch
from torch import nn
from torch.nn import Conv2d, ConvTranspose2d, LeakyReLU
from lightning_fabric.utilities.throughput import measure_flops

from speechlightning.models.modular_model import ModularModel

LIST_TYPE = list


def apply_complex_mask(signal: torch.Tensor, mask: torch.Tensor, manip: None, return_mask=False) -> torch.Tensor:
    """
    Multily a complex signal with a complex mask. Operates on the shapes (Real/Complex x Frequency x Time)
    :param signal: Signal to mask of shape (CxFxT)
    :param mask: Mast to apply on signal of shape (CxFxT)
    :return: Signal with applied mask of shape (CxFxT)
    """
    mask_complex_permuted = torch.view_as_complex(mask.permute((0, 2, 3, 1)).contiguous())
    input_complex_permuted = torch.view_as_complex(signal.permute((0, 2, 3, 1)).contiguous())

    if manip is not None:
        mask_complex_permuted = torch.polar(manip(mask_complex_permuted.abs()), mask_complex_permuted.angle())

    masked = mask_complex_permuted * input_complex_permuted
    output = torch.permute(torch.nan_to_num(torch.view_as_real(masked)), (0, 3, 1, 2)).contiguous()

    if return_mask:
        mask = torch.permute(torch.nan_to_num(torch.view_as_real(mask_complex_permuted)), (0, 3, 1, 2)).contiguous()
        return [output, mask]

    return [output]


class ChannelShuffle(torch.nn.Module):

    def __init__(self, shuffle_val, dim=1):
        super().__init__()

        self.forward = self.shuffle if shuffle_val > 1 else self.unit

        self.dim = dim
        self.shuffle_val = shuffle_val

    def shuffle(self, signal):

        tmp_dim = signal.shape[self.dim] // self.shuffle_val

        signal = signal.unflatten(self.dim, (tmp_dim, self.shuffle_val)).transpose(self.dim, self.dim+1).flatten(self.dim, self.dim+1)

        return signal

    def unit(self, signal):

        return signal


class PadCausal():
    def __init__(self, padding, c_axis=1) -> None:

        if isinstance(c_axis, int):
            c_axis = [c_axis]

        self.pad_vals = []
        for i, val in enumerate(padding):
            self.pad_vals.extend([0,val] if i in c_axis else [(val+1)//2, val//2])

        self.pad_vals = self.pad_vals[::-1]

    def __call__(self, signal):
        return torch.nn.functional.pad(signal, self.pad_vals)


class EncoderBlock(torch.nn.Module):
    """
    Encoder half of the EDBLock as used in the EffCRN with skip connection convolution
    """

    def __init__(self, in_channels_encoder, out_channels, kernel_size, stride,
                 skip_conv=True, skip_channels=None, residual_conv=False, batch_norm=False,
                 groups_high=1, groups_low=1, shuffle_high=True, shuffle_low=True):
        super().__init__()
        self.upper_encoder_conv = Conv2d(in_channels=in_channels_encoder, out_channels=out_channels,
                                         kernel_size=kernel_size, padding='same', groups=groups_high)
        self.lower_encoder_conv = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=(stride, 1), padding=(int((kernel_size[0]-1) / 2), 0) if stride > 1 else 'same', groups=groups_low)

        self.skip_conv = skip_conv
        self.res_conv = residual_conv
        self.batch_norm = batch_norm

        if skip_conv:
            skip_channels = skip_channels or out_channels
            self.skip_connection = Conv2d(in_channels=out_channels, out_channels=skip_channels, groups=math.gcd(skip_channels, out_channels),
                                        kernel_size=(1, 1))
        if residual_conv:
            self.residual_conv = Conv2d(in_channels=in_channels_encoder, out_channels=out_channels, groups=math.gcd(in_channels_encoder, out_channels),
                                        kernel_size=(1, 1), stride=(stride, 1))

        if batch_norm:
            self.bn_up      = nn.BatchNorm2d(num_features=out_channels)
            self.bn_down    = nn.BatchNorm2d(num_features=out_channels)

        self.lrelu = LeakyReLU(0.2)

        self.pad_causal = PadCausal([0, kernel_size[1]-1])

        self.mid_shuffle = ChannelShuffle(groups_high*shuffle_high)
        self.end_shuffle = ChannelShuffle(groups_low*shuffle_low)

    def forward(self, encoder_input):
        """
        Path down the encoder side of the EDBlock
        :param encoder_input: Encoder input as (in_channels_encoder x frequency x time)
        :return: encoder output as (out_channels, frequeny/stride, time), skip connection as (out_channels, frequency, time)
        """
        encoder_intermediate = self.mid_shuffle(self.lrelu(self.upper_encoder_conv(self.pad_causal(encoder_input))))
        if self.batch_norm:
            encoder_intermediate = self.bn_up(encoder_intermediate)
        encoder_output = self.lrelu(self.lower_encoder_conv(self.pad_causal(encoder_intermediate)))
        if self.batch_norm:
            encoder_output = self.bn_down(encoder_output)

        skip_connection_output = self.skip_connection(encoder_intermediate) if self.skip_conv else encoder_intermediate
        if self.res_conv:
            encoder_output = encoder_output + self.residual_conv(encoder_input)

        return self.end_shuffle(encoder_output), skip_connection_output


class DecoderBlock(torch.nn.Module):
    """
    Decoder half of the EDBlock as used in the CGGN without skip connection convolution
    """

    def __init__(self, in_channels_decoder, out_channels, kernel_size, stride,
                 use_skip=True, residual_conv=False, batch_norm=False,
                 groups_high=1, groups_low=1, shuffle_high=True, shuffle_low=True):
        super().__init__()
        self.upper_decoder_conv = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         padding='same', groups=groups_high)
        self.lower_decoder_conv = ConvTranspose2d(in_channels=in_channels_decoder, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=(stride, 1),
                                                  padding=(int(kernel_size[0] / 2-1), 0) if stride > 1 else (1,0), groups=groups_low)
        self.lrelu = LeakyReLU(0.2)

        self.use_skip = use_skip
        self.res_conv = residual_conv
        self.batch_norm = batch_norm

        if residual_conv:
            self.residual_conv = ConvTranspose2d(in_channels=in_channels_decoder, out_channels=out_channels, groups=math.gcd(in_channels_decoder, out_channels),
                                        kernel_size=(1, 1), output_padding=(int(kernel_size[0] / 2), 0), stride=(stride, 1))
        if batch_norm:
            self.bn_up      = nn.BatchNorm2d(num_features=out_channels)
            self.bn_down    = nn.BatchNorm2d(num_features=out_channels)

        self.mid_shuffle = ChannelShuffle(groups_low*shuffle_low)
        self.end_shuffle = ChannelShuffle(groups_high*shuffle_high)

    def forward(self, decoder_input, skip_connection_input=None):
        """
        Path up the decoder side of the EDBlock
        :param decoder_input: Decoder input as (in_channels_decoder x frequency x time)
        :param encoder_intermediate: skip connection as (out_channels, frequency * stride, time)
        :return: decoder output as (out_channels, frequency * stride, time)
        """
        out_size = decoder_input.size(dim=2)*2
        decoder_intermediate = self.mid_shuffle(self.lrelu(self.lower_decoder_conv(decoder_input)[:,:,:out_size,:]))
        if self.batch_norm:
            decoder_intermediate = self.bn_down(decoder_intermediate)
        if self.use_skip:
            decoder_intermediate = decoder_intermediate + skip_connection_input
        decoder_output = self.lrelu(self.upper_decoder_conv(decoder_intermediate))
        if self.batch_norm:
            decoder_output = self.bn_up(decoder_output)
        if self.res_conv:
            decoder_output = decoder_output + self.residual_conv(decoder_input)
        return self.end_shuffle(decoder_output)


class Output(NamedTuple):
    out: torch.Tensor
    mask: Optional[torch.Tensor]


class gGCRN_arch(nn.Module):
    def __init__(self, n_feature, skips, res_convs, batch_norm, inp_compression_factor=1.0, comp_tech='model_in',
                 groups_high=1, groups_low=1, shuffle_high=True, shuffle_low=True, gru_skip=False,
                 number_ed_blocks=3, kernel_N=3, featuremap_F=40, featuremap_G=None, N_split=10,
                 rec_strat='GRU', stride=2,
                 num_in=2, rt_mode:bool = False, output_type='mask', return_aux=False, return_int_layers=False):
        """
        Main CGGN16 architecture.
        :param n_feature: expected feature count (N//2+1)
        :param num_in: number of inputs (signals, not channels)
        :param rt_mode: Whether to run in realtime (frame-wise processing) or offline.
        :param output_type: 'mask', 'echo', or 'mapping'
        :param return_aux: Whtether to return the estimate (for mask or echo) as 2nd output.
        """
        super().__init__()
        self.number_ed_blocks = number_ed_blocks
        self.rt_mode = rt_mode
        self.n_feature = n_feature

        self.in_compo_fac = inp_compression_factor
        self.comp_tech = comp_tech if inp_compression_factor != 0 else 'none'

        try:
            assert output_type in ['mask', 'echo', 'mapping']
            self.output_type = output_type
        except AssertionError:
            raise ValueError(f'Output type must be mask, echo, or mapping. Received {output_type}.')
        except:
            raise TypeError('Failed to initialize output type. Check argument.')

        try:
            assert comp_tech in ['model_in', 'io', 'model', 'none', 'log']
        except AssertionError:
            raise ValueError(f'Compression technique must be model_in, model, or io. Received {comp_tech}.')
        except:
            raise TypeError('Failed to initialize output type. Check argument.')

        self.return_aux = return_aux
        self.return_int_layers = return_int_layers

        self.encoder_blocks = torch.nn.ModuleList()
        self.decoder_blocks = torch.nn.ModuleList()
        self.GRU_splits     = torch.nn.ModuleList()

        kernel_size = (kernel_N, 1)

        featuremap_F = [featuremap_F*i for i in range(1,number_ed_blocks+1)]*2 if isinstance(featuremap_F, int) else featuremap_F
        featuremap_F = [featuremap_F]*2 if isinstance(featuremap_F[0], int) else featuremap_F

        featuremap_G = featuremap_G or featuremap_F[0][0]

        #BN reduction coefficient
        red_coeff = int(np.prod(stride[0]))

        self.N_pad = n_feature % red_coeff
        if self.N_pad > 0:
            self.N_pad = red_coeff - self.N_pad
        self.N_split = N_split
        N_bott = (n_feature+self.N_pad)//red_coeff
        self.N_units_GRU = N_bott*featuremap_G//(self.N_split)

        for i in range(self.number_ed_blocks):
            encoder_block = EncoderBlock(
                in_channels_encoder=num_in*2 if i == 0 else featuremap_F[0][i-1],
                out_channels=featuremap_F[0][i],
                kernel_size=kernel_size,
                stride=stride[0][i],
                skip_conv=skips[0][i],
                skip_channels=featuremap_F[1][i],
                residual_conv=res_convs[0][i],
                groups_high=groups_high[0][i],
                groups_low=groups_low[0][i],
                shuffle_high=shuffle_high[0][i],
                shuffle_low=shuffle_low[0][i],
                batch_norm=batch_norm
            )
            decoder_block = DecoderBlock(
                in_channels_decoder=featuremap_F[1][i] if i == self.number_ed_blocks - 1 else featuremap_F[1][i+1],
                out_channels=featuremap_F[1][i],
                kernel_size=kernel_size,
                stride=stride[1][i],
                use_skip=skips[1][i],
                residual_conv=res_convs[1][i],
                groups_high=groups_high[1][i],
                groups_low=groups_low[1][i],
                shuffle_high=shuffle_high[1][i],
                shuffle_low=shuffle_low[1][i],
                batch_norm=batch_norm
            )
            self.encoder_blocks.append(encoder_block)
            self.decoder_blocks.append((decoder_block))

        conv_strat = nn.Conv2d

        if rt_mode:
            rec_strat = nn.GRUCell

        else:
            if rec_strat == "GRU":
                rec_layer = nn.GRU
            elif rec_strat == "LSTM":
                rec_layer = nn.LSTM

        self.init_GRU: List[Optional[torch.Tensor]] = [None] * self.N_split

        self.bottleneck_conv_in = conv_strat(in_channels=featuremap_F[0][-1], out_channels=featuremap_G,
                                    kernel_size=kernel_size, padding='same')
        self.bottleneck_conv_out = conv_strat(in_channels=featuremap_G, out_channels=featuremap_F[1][-1],
                                    kernel_size=kernel_size, padding='same')

        for i in range(self.N_split):
            self.GRU_splits.append(rec_layer(input_size=self.N_units_GRU, hidden_size=self.N_units_GRU))

        self.gru_skip = gru_skip

        if gru_skip:
            self.gru_skip_conv = conv_strat(in_channels=featuremap_G, out_channels=featuremap_G, groups=featuremap_G,
                                        kernel_size=(1, 1))

        self.post_conv = conv_strat(in_channels=featuremap_F[1][0], out_channels=2, kernel_size=kernel_size, padding='same')

        self.unflatten = torch.nn.Unflatten(-1, (featuremap_G, N_bott))
        self.l_relu = LeakyReLU(0.2)
        self.manip = nn.Tanh()

    def forward(self, signals, hidden=None) -> Output:

        mic_stft = signals[0]

        if self.comp_tech != 'none':
            #apply compression to complex
            signals = [torch.view_as_complex(sig.transpose(1,-1).contiguous()) for sig in signals]
            if self.comp_tech == 'log':
                signals = [torch.view_as_real(torch.polar(torch.log10(torch.abs(sig)+torch.finfo(sig.dtype).eps), torch.angle(sig))).transpose(1,-1) for sig in signals]
            else:
                signals = [torch.view_as_real(torch.polar(torch.pow(torch.abs(sig), self.in_compo_fac), torch.angle(sig))).transpose(1,-1) for sig in signals]

        if self.comp_tech == 'io':
            mic_stft = signals[0]

        input_stack = torch.concatenate(signals, dim=1)

        encoder_outputs: List[Optional[torch.Tensor]] = [None] * self.number_ed_blocks
        skips = [None] * self.number_ed_blocks
        for n, encoder_block in enumerate(self.encoder_blocks):
            encoder_outputs[n], skips[n] = encoder_block.forward(
                input_stack if n == 0 else encoder_outputs[n - 1]
            )

        bottleneck_conv_in_output: torch.Tensor = self.l_relu(self.bottleneck_conv_in(encoder_outputs[-1]))

        if self.rt_mode:
            bottleneck_conv_in_output = bottleneck_conv_in_output.squeeze(-1)
        else:
            bottleneck_conv_in_output = bottleneck_conv_in_output.permute(3, 0, 1, 2) # [B, C, H, W] -> [W, B, C, H]

        gru_input = torch.flatten(bottleneck_conv_in_output, -2, -1)  #[(W,) B, C, H] -> [(W,) B, C*H]

        splits = torch.split(gru_input, self.N_units_GRU, -1)
        GRU_outputs: List[Optional[torch.Tensor]] = [None] * self.N_split
        for n, GRU_branch in enumerate(self.GRU_splits):
            if self.rt_mode:
                GRU_outputs[n] = GRU_branch(splits[n], hidden[n])
            else:
                GRU_outputs[n], _ = GRU_branch(splits[n])
        GRU_out = torch.concatenate(GRU_outputs, dim=-1)

        reshaped_gru_output = self.unflatten(GRU_out) # [(W,) B, C*H] -> [(W,) B, C, H]

        if self.rt_mode:
            reshaped_gru_output = reshaped_gru_output.unsqueeze(-1)
        else:
            reshaped_gru_output = reshaped_gru_output.permute(1, 2, 3, 0)  # [W, B, C, H] -> [B, C, H, W]

        if self.gru_skip:
            reshaped_gru_output = reshaped_gru_output + self.gru_skip_conv(reshaped_gru_output)

        bottleneck_conv_out_output: torch.Tensor = self.l_relu(self.bottleneck_conv_out(reshaped_gru_output))

        decoder_outputs: List[Optional[torch.Tensor]] = [None] * self.number_ed_blocks
        for n, decoder_block in reversed(list(enumerate(self.decoder_blocks))):
            decoder_block: DecoderBlock = decoder_block
            decoder_outputs[n] = decoder_block.forward(
                decoder_input=bottleneck_conv_out_output if n == self.number_ed_blocks - 1 else decoder_outputs[n + 1],
                skip_connection_input=skips[n]
            )

        post_conv_output = self.post_conv(decoder_outputs[0])

        if self.comp_tech == 'model':
            post_conv_output = torch.view_as_complex(post_conv_output.transpose(1,-1).contiguous())
            post_conv_output = torch.view_as_real(torch.polar(torch.pow(torch.abs(post_conv_output), 1/self.in_compo_fac), torch.angle(post_conv_output))).transpose(1,-1)

        if self.output_type == 'mask':
            output = apply_complex_mask(mic_stft, post_conv_output, self.manip, self.return_aux)
        elif self.output_type == 'echo':
            output = [mic_stft - post_conv_output]
            if self.return_aux:
                output.append(post_conv_output)
        else: # mapping
            output = [post_conv_output]

        if self.comp_tech == 'io':
            output = [torch.view_as_complex(sig.transpose(1,-1).contiguous()) for sig in output]
            output = [torch.view_as_real(torch.polar(torch.pow(torch.abs(sig), 1/self.in_compo_fac), torch.angle(sig))).transpose(1,-1) for sig in output]

        if self.return_int_layers:
            output.append([encoder_outputs[-1], bottleneck_conv_out_output])

        if self.rt_mode:
            return output, GRU_outputs
        else:
            return output


class CGGN16_AEC_model(ModularModel):
    """
    Baseline CGGN16 model for AEC (multi I/O design), cf. Seidel, 2022. DOI: 10.1109/IWAENC53105.2022.9914758
    """

    def __init__(self, cfg, *args, num_in=2, inp_compression_factor=1.0, compression_technique='model_in',
                 skips=True, res_convs=False, gru_skip=False, batch_norm=False,
                 groups_high=1, groups_low=1, shuffle_high=True, shuffle_low=True,
                 number_ed_blocks=3, kernel_N=3, featuremap_F=40, featuremap_G=None, N_split=10,
                 rec_strat='GRU', stride=2,
                 rt_mode=False, output_type='mask', return_aux=False, return_int_layers=False, **kwargs):
        """
        :param cfg: Config dict containing framing parameters.
        :param num_in: number of inputs (signals, not channels)
        :param rt_mode: Whether to run in realtime (frame-wise processing) or offline.
        :param output_type: 'mask', 'echo', or 'mapping'
        :param return_aux: Whether to return the estimate (for mask or echo) as 2nd output.
        """
        super().__init__(*args, **kwargs)
        self.fb_precision = 'float64'

        self.rt_mode = rt_mode
        self.return_aux = return_aux
        self.return_int_layers = return_int_layers

        self.groups_high = groups_high
        self.groups_low = groups_low
        self.shuffle_high = shuffle_high
        self.shuffle_low = shuffle_low
        self.skips = skips
        self.stride = stride
        self.res_convs = res_convs

        for key in ['groups_high', 'groups_low', 'shuffle_high', 'shuffle_low', 'skips', 'stride', 'res_convs']:
            val = getattr(self, key)
            if isinstance(val, LIST_TYPE):
                setattr(self, key, [ival if isinstance(ival, LIST_TYPE) else [ival] * number_ed_blocks for ival in val])
            else:
                setattr(self, key, [[val] * number_ed_blocks, [val] * number_ed_blocks])

        #BN reduction coefficient
        red_coeff = int(np.prod(self.stride[0]))

        n_feature = cfg['n_fft'] // 2 + 1
        self.n_feature = n_feature
        self.N_pad = n_feature % red_coeff
        if self.N_pad > 0:
            self.N_pad = red_coeff - self.N_pad

        self.AEC_module = gGCRN_arch(n_feature=n_feature, skips=self.skips, res_convs=self.res_convs, batch_norm=batch_norm,
                                     num_in=num_in, inp_compression_factor=inp_compression_factor, comp_tech=compression_technique,
                                     groups_high=self.groups_high, groups_low=self.groups_low,
                                     shuffle_high=self.shuffle_high, shuffle_low=self.shuffle_low,
                                     gru_skip=gru_skip, number_ed_blocks=number_ed_blocks, kernel_N=kernel_N,
                                     featuremap_F=featuremap_F, featuremap_G=featuremap_G or featuremap_F, N_split=N_split,
                                     rec_strat=rec_strat, stride=self.stride,
                                     rt_mode=rt_mode, output_type=output_type, return_aux=return_aux, return_int_layers=return_int_layers)

        if rt_mode:
            self.hidden = self.main_module.init_GRU

        if not self.is_in_inference_mode:
            x = [torch.view_as_real(torch.randn(1, n_feature+self.N_pad, cfg['sampling_rate']//cfg['hop_size'],
                                                dtype=torch.complex64)).permute(0,3,1,2)] * num_in
            model_fwd = lambda: self.AEC_module(x) # noqa: E731
            fwd_flops = measure_flops(self.AEC_module, model_fwd)

            print('--------------------------------')
            print('FLOPS (G): ' + str(fwd_flops/1e9))
            print('--------------------------------')

    def pad_input(self, input: torch.Tensor):
        return torch.nn.functional.pad(input, (0, 0, self.N_pad, 0))

    def unpad_output(self, input: torch.Tensor):
        return input[:, :, self.N_pad:, :]

    def forward_frame(self, input_batch, meta_data=None):

        mic_stft = self.pad_input(torch.view_as_real(input_batch[0]).permute([0,2,1]).unsqueeze(1)).squeeze(1)
        farend_stft = self.pad_input(torch.view_as_real(input_batch[1]).permute([0,2,1]).unsqueeze(1)).squeeze(1)

        output, self.hidden = self.AEC_module(mic_stft, farend_stft, self.hidden)

        output = self.unpad_output(output.unsqueeze(3)).squeeze(3).permute(1,0,2)
        output = torch.complex(output[0],output[1])

        return output

    def forward(self, input_batch, meta_data=None) -> Output:

        in_sigs = [self.pad_input(torch.view_as_real(sig[:,:self.n_feature]).permute([0,3,1,2])) for sig in input_batch]

        # AEC
        if self.rt_mode:
            hidden = self.hidden
            output = torch.zeros_like(in_sigs[0])
            if self.return_aux:
                output_aux = torch.zeros_like(in_sigs[0])

            for t in range(in_sigs[0].shape[-1]):
                inp = ([sig[:,:,:,t:t+1] for sig in in_sigs], hidden)
                out_tmp, hidden = self.AEC_module(*inp)
                if self.return_aux:
                    output[:,:,:,t:t+1], output_aux[:,:,:,t:t+1] = out_tmp
                else:
                    output[:,:,:,t:t+1] = out_tmp
        else:
            out_tmp = self.AEC_module(in_sigs)

            if self.return_int_layers:
                int_layers = out_tmp.pop(-1)

            output = out_tmp

        #DIM: (B x 2 x Nfft x T)

        output = [self.unpad_output(sig).permute(1,0,2,3) for sig in output]
        output = [torch.complex(sig[0],sig[1]) for sig in output]

        if self.return_int_layers:
            output.extend(int_layers)

        return output
