%%%IR GEN
function [ir] = IR_GEN(length, fs, seed, tsixty)

x=1:9/length:10;
stream = RandStream('mt19937ar','Seed', seed);
RandStream.setGlobalStream(stream);
w = randn(length, 1);
t60 = -1;
expo = 3;
it = 0;

while (t60 < tsixty-0.001 || t60 > tsixty+0.001) && it < 50
	if t60 < tsixty-0.001
		expo = expo-tsixty;
	else 
		expo = expo+tsixty;
	end
	EDC=exp(-x.^expo);

	wdecay = w.*EDC(1:end-1).';


	t60 = reverb(wdecay, fs);
	it = it+1;
end
ir = wdecay.';
% it
% expo
% t60