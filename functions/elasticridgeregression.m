function  [ iesn ] = elasticridgeregression( X, Y, esn)
	iesn = esn;
	lambda = esn.lambda;

	if lambda == 0.
		W = Y'*X'/(X*X'+lambda*eye(size(X,1)));
	else
		while true
			try	
				W = Y'*X'/(X*X'+lambda*eye(size(X,1)));
    			break
			catch
				lambda = lambda*2.;
			end
		end
	end

	iesn.lambda = lambda;
	iesn.Wout = W;
end
