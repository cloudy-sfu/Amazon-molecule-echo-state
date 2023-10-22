function  [ gc, lambda ] = GCx(data, Nr, leakRate, spectralRadius, ...
    inputScaling, reg, nonlinearfunction, washout)

arguments
    data
    Nr double = 25
    leakRate double = 0.6
    spectralRadius double = 0.9
    inputScaling double = 0
    reg double = 1e-8
    nonlinearfunction function_handle = @tanh  
    % tanh 
    % relu
    % relog: y = [0, x <= 0; log(x + 1), x > 0]
    % logan: y = sign(x) .* log(abs(x)+1)
    % iden: y = x
    washout int32 = 0  % recommended: 0.005*len
end

[~, nodes]= size(data);

Wr=[];
for i=1:nodes
	wwtemp = rand(Nr,Nr); 
    wwtemp = spectralRadius * orth(wwtemp);
	Wr = blkdiag(Wr,wwtemp);
end
Win = []; 
for i=1:nodes
	wtemp = inputScaling*(rand(Nr,1) * 2 - 1); 
	Win=blkdiag(Win,wtemp); 
end

errors=zeros(nodes,1);

trY = data(2+washout:end,:);
trX = data(1:end-1,:);

%% UNRESTRICTED MODELS
esnAll = iESN( ...
    Nr*ones(1,nodes), nodes, Wr, Win, leakRate, spectralRadius, ...
    1, 0, reg, 1, 1, nonlinearfunction ...
);
esnAll.train(trX,trY,washout);
output = esnAll.predict(trX,washout);
for jj=1:nodes  
    errors(jj)= mean_squared_error(output(1:end,jj), trY(1:end,jj)); 
end

%% RESTRICTED MODELS
errors2=zeros(nodes,nodes);
gc=zeros(nodes,nodes);
esn = cell(nodes, 1);
for j = 1 : nodes
    trXY = data(1:end-1,(1:nodes) ~= j);
 	trYY = trY(1:end,(1:nodes) ~= j);
	wtemp=esnAll.Wr(kron(1:nodes,ones(1,Nr))~=j,kron(1:nodes,ones(1,Nr))~=j);
	wintemp=Win(kron(1:nodes,ones(1,Nr))~=j,(1:nodes) ~= j);
	esn{j} = iESN( ...
        Nr*ones(1,nodes-1), nodes-1, wtemp, wintemp, leakRate, spectralRadius, ...
        1, 0, esnAll.lambda, 1, 1, nonlinearfunction ...
    );

	esn{j}.train(trXY,trYY,washout);
	output2 = esn{j}.predict(trXY,washout);
    for i = 1:nodes
		if (i==j)
			gc(j,i)= nan;
			errors2(i,j)= nan;   % pleonastico
		elseif (i<j) 
			errors2(i,j)= mean_squared_error(output2(1:end,i),trYY(1:end,i));
			gc(j,i) = log(errors2(i,j)/errors(i));
		else	
			errors2(i,j)= mean_squared_error(output2(1:end,i-1),trYY(1:end,i-1));
			gc(j,i) = log(errors2(i,j)/errors(i));
		end
    end
	lambda = esnAll.lambda;
end


