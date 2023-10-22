%%%%%	This is a modified implementation initially taken from https://github.com/stefanonardo/echo-state-network/

classdef iESN < handle
    properties
        Nr 
        Ntot 
        alpha
        rho
        inputScaling
        biasScaling
        lambda
        connectivity
        orthonormalWeights
	    seqDim
        Win
        Wb
        Wr
        Wout
        internalState
	    nonlinearfunction    
	    wrexternallydefined    
	    winexternallydefined    
    end
    methods
        function iesn = iESN(Nr, seqDim, Wr, Win, leakRate, ...
                spectralRadius, inputScaling, biasScaling, regularization, ...
                connectivity, orthonormalWeights, nonlinearfunction ...
        )
            arguments
                Nr  % reservoir's size
                seqDim
                Wr
                Win
                leakRate double = 1  % leakage rate
                spectralRadius double = 0.9  % spectral radius
                inputScaling double = 1  % input weights scale 
                biasScaling double = 0  % bias weights scale 
                regularization double = 1  % regularization parameter
                connectivity double = 1  % reservoir connectivity
                orthonormalWeights double = 1
                nonlinearfunction function_handle = @tanh
            end
        	    
        	iesn.Nr = Nr;
		    iesn.Ntot = sum(Nr);
        	iesn.alpha = leakRate;
        	iesn.rho = spectralRadius;
        	iesn.inputScaling = inputScaling;
        	iesn.biasScaling = biasScaling;
        	iesn.lambda = regularization;
        	iesn.connectivity = connectivity;
        	iesn.orthonormalWeights = orthonormalWeights;
        	iesn.nonlinearfunction = nonlinearfunction;
        	iesn.wrexternallydefined = 0;
        	iesn.winexternallydefined = 0;
            iesn.Wr = Wr;
            iesn.Win = Win;
            iesn.seqDim = seqDim;
            
            assert(seqDim == size(Nr, 2), "Sequence dimension mismatch.");
		    if ~(iesn.winexternallydefined) 
    		    iesn.Win = []; 
                for i=1:seqDim 
                    wtemp = iesn.inputScaling*(rand(Nr(i),1) * 2 - 1); 
                    iesn.Win=blkdiag(iesn.Win,wtemp); 
                end
		    end
		    iesn.Wb = iesn.biasScaling * (rand(iesn.Ntot, 1) * 2 - 1);

		    if ~(iesn.wrexternallydefined) 
			    if(iesn.orthonormalWeights)
				    Wr=[];
                    for i=1:seqDim
					    wwtemp = rand(Nr(i),Nr(i)); 
                        wwtemp = iesn.rho * orth(wwtemp);
					    Wr = blkdiag(Wr,wwtemp);
                    end
			    else
        		    Wr = full(sprand(iesn.Ntot-size(iesn.Wr,1),iesn.Ntot-size(iesn.Wr,1), iesn.connectivity));
		            Wr(Wr ~= 0) = Wr(Wr ~= 0) * 2 - 1;
        		    Wr = Wr * (iesn.rho / max(abs(eig(Wr))));
				    Wr = blkdiag(zeros(size(iesn.Wr,1),size(iesn.Wr,1)),Wr);
			    end
			    iesn.Wr = blkdiag(iesn.Wr , Wr(1+size(iesn.Wr,1):end,1+size(iesn.Wr,1):end));
		    end

        end
        function train(iesn, trX, trY, washout, varargin)
            % Trains the network on input X given target Y.
            %
            % args: 
            %   trX: cell array of size N x 1 time series. Each cell contains an
            %   array of size sequenceLenght x sequenceDimension.
            %   trY: target matrix composed by all sequences. Washout must be 
            %   applied before calling this function.
            %   washout: number of initial timesteps not to collect.

            assert(iesn.seqDim == size(trX, 2), "Sequence dimension mismatch.");
            trainLen = size(trY,1);

  	        numvarargs = length(varargin);
  	        for i = 1:2:numvarargs
  	            switch varargin{i}
  	                case 'XX', X = varargin{i+1};
  	                otherwise, error('the option does not exist');
  	            end
            end
            if exist('X','var') ~= 1
                X = zeros(1+iesn.seqDim+iesn.Ntot, trainLen);
                idx = 1;
                U = trX';
                x = zeros(iesn.Ntot,1);
                for i = 1:size(U,2)
                    u = U(:,i);
                    x_ = iesn.nonlinearfunction(iesn.Win*u + iesn.Wr*x + iesn.Wb);
                    x = (1-iesn.alpha)*x + iesn.alpha*x_;
                    if i > washout
                        X(:,idx) = [1;u;x];
                        idx = idx+1;
                    end
                end
            end

            iesn.internalState = X(1+iesn.seqDim+1:end,:);
            iesn = elasticridgeregression(X, trY, iesn);
        end
        function y = predict(iesn, data, washout)
            % Computes the output given the data.
            %
            % args:
            %   data: cell array of size N x 1 time series. Each cell contains an
            %   array of size sequenceLenght x sequenceDimension.
            %   washout: number of initial timesteps to not collect.
            %
            % returns:
            %   y: predicted output.
            
            iesn.seqDim = size(data,2);
            trainLen = size(data, 1) - washout;
            
            X = zeros(1+iesn.seqDim+iesn.Ntot, trainLen);
            idx = 1;
            U = data';
            x = zeros(iesn.Ntot,1);
            
            for i = 1:size(U,2)
                u = U(:,i);
                x_ = iesn.nonlinearfunction(iesn.Win*u + iesn.Wr*x + iesn.Wb); 
                x = (1-iesn.alpha)*x + iesn.alpha*x_;
                if i > washout
                    X(:,idx) = [1;u;x];
                    idx = idx+1;
                end
            end
            
            iesn.internalState = X(1+iesn.seqDim+1:end,:);
            y = iesn.Wout*X;
            y = y';
        end
    end
end
