%%% This code is for "IRS-based Wireless Jamming Attacks: How Jammers can Attack without Power?"
%%%
%%% Code modified by André Gomes, gomesa@vt.edu.

% clear;
% parameter settings

% the coordinates of the LT, the LR, and the IRS
% x_lt = 0;
% y_lt =0;
% x_lr = 10;
% y_lr = 0;
% x_irs = 5;
% y_irs = 2;

% the distances between these devices
% d_ti = sqrt((x_irs - x_lt)^2 + (y_irs - y_lt)^2 ); % the distance between the LT and the IRS;
% d_tr = sqrt( (x_lr - x_lt)^2 + (y_lr - y_lt)^2 ); % the distance between the LT and the LR;
% d_ir = sqrt( (x_lr - x_irs)^2 + (y_lr - y_irs)^2); % the distance between the IR and the IRS.

% alphaLTLR = 3.5; % the path-loss exponent for the link between the LT and LR
% alphaIRS = 2.8; % the path-loss exponent for the links connected to the IRS
% M = 8; % the number of anteena at the LT;
% N = 150; % the number of passive reflecting elements
% % Count = 50; % the number of channel generations
% Iteration = 1e3; % the number of Gaussian randomization method
% PdBmSet = 10:5:40;
% PSet = 1e-3 .* 10.^(PdBmSet./10);

% PadBm1 = 15; % dBm the active transmit power of the jammer
% Pa1 = 1e-3 * 10^(PadBm1 /10);
% PadBm2 = 25; % dBm the active transmit power of the jammer
% Pa2 = 1e-3 * 10^(PadBm2 /10);

% PadBm3 = 35; % dBm the active transmit power of the jammer
% Pa3 = 1e-3 * 10^(PadBm3 /10);
% alphaPa = 0.5;  % the attenuation factor for the active jamming scheme

% Noise = 1e-9; % -60 dBm

% b = 5; % the phase resolution in number of bits
% L = 2^b;
% thetaSet = 0: 2*pi/L: 2*pi*(L-1)/L;

% SNRPowerProAve = zeros(1,length(PSet));

% SNRPowerWoIRSAve = zeros(1,length(PSet));
% 
% SINRPowerActiveJammingAve1 = zeros(1,length(PSet));
% SINRPowerActiveJammingAve2 = zeros(1,length(PSet));
% SINRPowerActiveJammingAve3 = zeros(1,length(PSet));

% PSet = 1e-3 .* 10.^(30/10); % Test a single example of 30 dBm Tx power.

function [Gamma, Theta] = optimizeRIS(PdBm, w_bs2ue, w_bs2ris, M, N, Cd, Cr, G, b)
    %% Hyperparameters.
    Iteration = 1e3; % the number of Gaussian randomization method
    % b = 5; % the phase resolution in number of bits
    L = 2^b;
    thetaStep = 2*pi/L;
    thetaSet = 0: thetaStep: 2*pi*(L-1)/L;
    
    P = 1e-3 * 10^(PdBm/10);
% for j =1: length(PSet)
%     j
%     P = PSet(j);
    Count = 1; % Number of channel measurements. TODO: get rid of the for loop that depends on Count.
%     SNRPowerWoIRSSet = zeros(Count, 1);
    
%     SINRActiveJammingSet1 = zeros(Count,1);
%     SINRActiveJammingSet2 = zeros(Count,1);
%     SINRActiveJammingSet3 = zeros(Count,1);
    
%     SNRPowerProSet = zeros(Count,1);
    
    for i =1:Count
%         i
        % channel generation
%         Cd = sqrt(1e-3 * d_tr^(-alphaLTLR) /2 ) * (randn(M,1) + 1i * randn(M,1)); % the channel between the LT and IR
%         Cr = sqrt(1e-3 *  d_ir^(-alphaIRS) /2 ) * (randn(N,1) + 1i * randn(N,1)); % the channel between the IRS and IR
%         G = sqrt(1e-3 * d_ti^(-alphaIRS) /2 ) * (randn(N,M) + 1i * randn(N,M) ); % the channel between the LT adn IRS
         
        % Omega = sqrt(double(P/M)) * ones(M,1); % linear beamforming vector
        
        
        %% proposed scheme
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%% Solve P2 with the given amplitude reflection coefficients  %%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        beta = ones(N,1); % the amplitude reflection coefficient
        Gamma = diag(beta);
        A = diag(Cr' * Gamma) * G * w_bs2ris;
        psi = Cd' * w_bs2ue;
        % disp(["Psi = ", psi])
        
        R = [ A * A',  A * psi';  A' * psi, 0];

        cvx_precision high

        cvx_begin quiet
        variable V(N+1,N+1) complex semidefinite
        
        minimize real(trace(R * V)) + abs(psi)^2
        subject to
        for nn = 1:N+1
            V(nn,nn) == 1;
        end
        cvx_end
        
        
        
        % compute the SVD of V
        [U1,S,U2] = svd(V); % SVD operation
        
        % the Gaussian randomization method
        vBarSetDis = zeros(N,Iteration);
        %vBarSetCon = zeros(N, Iteration);
        RecePowerSet  = zeros(Iteration,1);
        rand("twister", 0);
        for ii =1:Iteration
            % refer to Eq. 3;
            muBar = U1 * sqrt(S) * (sqrt(1/2) * norminv(rand(N+1,1), 0, 1) + sqrt(1/2) * 1i * norminv(rand(N+1, 1), 0, 1));
            % refer to Eq. 4;
            vtemp = muBar ./ muBar(N+1);
            thetaObtained = - angle(vtemp(1:N));
            
            % selected the optimal phase shifts from the set of discrete values
            thetaDis = zeros(N,1);
            for kk = 1:N
                thetaTemp = thetaObtained(kk);
                % transform the phase shifts into the range of [0,2*pi].
                while thetaTemp < 0
                    thetaTemp = thetaTemp + 2*pi;
                end
                while thetaTemp > 2*pi
                    thetaTemp = thetaTemp - 2*pi;
                end
                thetaNew = thetaTemp;
                % Border case.
                if 2*pi - thetaNew < thetaStep/2
                    index = 1;
                % General cases.
                else
                    diff = abs(thetaNew - thetaSet);
                    [~,index] = min(diff);
                end
                thetaDis(kk) = thetaSet(index);
            end
            
            vBarSetDis(:,ii) = conj(exp(1j * thetaDis ));
            RecePowerSetDis(ii,1) = real(vBarSetDis(:,ii)' * A * A' * vBarSetDis(:,ii) + vBarSetDis(:,ii)' * A * psi' + psi * A' * vBarSetDis(:,ii) + abs(psi)^2); 
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%% Solve P3 with the given phase shifts %%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        
        [~,index] = min(RecePowerSetDis);
        vBarBest = conj(vBarSetDis(:,index)); % optimal thetaDis, exp(1j * OptimalthetaDis) 
        % [~,index] = min(RecePowerSetCon);
        % vBarBestCon = conj(vBarSetCon(:,index));
        % Uncomment line below to bypass the optimization of the phase coefficients.
        % vBarBest(:) = 0;



        C = diag(Cr') * diag(vBarBest) * G * w_bs2ris;

        
        cvx_begin quiet
           variable betaCVX(N,1) nonnegative 
           minimize norm(betaCVX' * C + psi)
           subject to 
               betaCVX - ones(N,1) <=0;
         cvx_end


        %% Solve optimization without quantization.

        %CCon = diag(Cr') * diag(vBarBestCon) * G * w_bs2ris;

        %cvx_begin quiet
        %   variable betaCVXCon(N,1) nonnegative 
        %   minimize norm(betaCVXCon' * CCon + psi)
        %   subject to 
        %       betaCVXCon - ones(N,1) <=0;
        % cvx_end
        % Uncomment line below to bypass the optimization of the magnitude coefficients.
        % betaCVX(:) = 1;
        
        %% RETURN values.
        Gamma = betaCVX ;
        Theta = vBarBest;
        %GammaCon = betaCVXCon;
        %ThetaCon = vBarBestCon;

        % A = Cr' * diag(Gamma); 
        % B = A * diag(Theta);
        % C = B * G;
        % D = C * Omega;
        % disp(["A = ", A(1,1)]);
        % disp(["B = ", B(1,1)]);
        % disp(["C = ", C(1,1)]);
        % disp(["D = ", D(1,1)]);
        % prx = norm(D + psi)^2; 
        % disp(["prx = ", prx])

        
        % RecePowerProSet = norm(betaCVX' * C + psi)^2;
%         SNRPowerProSet(i) = RecePowerProSet / Noise;
        
        %%  active jamming scheme

%         ReActive1 = abs(sqrt(Pa1/N) * Cr' * ones(N,1))^2;
   
%         SINRActiveJammingSet1(i,1) = abs(psi).^2 / (ReActive1 + Noise);
        
%         ReActive2 = abs(sqrt(Pa2/N) * Cr' * ones(N,1))^2;
%         SINRActiveJammingSet2(i,1) = abs(psi).^2 / (ReActive2 + Noise);
        
%         ReActive3 = abs(sqrt(Pa3/N) * Cr' * ones(N,1))^2;
%         SINRActiveJammingSet3(i,1) = abs(psi).^2 / (ReActive3 + Noise);
        
        %% without IRS
%         RecePowerWoIRSSet = abs(psi).^2;
%         SNRPowerWoIRSSet(i,1) = RecePowerWoIRSSet / Noise;
        
%         PdBm = 10*log10(P/1e-3);
% 
%         beta_fname = "beta_" + PdBm + "dBm_" + i + ".mat";
%         theta_fname = "theta_" + PdBm + "dBm_" + i + ".mat";
%         
%         save(beta_fname, "betaCVX");
%         save(theta_fname, "vBarBest");
        
    end
    
%     SNRPowerProAve(j) = sum(SNRPowerProSet) / (Count);
    
%     SINRPowerActiveJammingAve1(j) = sum(SINRActiveJammingSet1) / (Count);
%     SINRPowerActiveJammingAve2(j) = sum(SINRActiveJammingSet2) / (Count);
%     SINRPowerActiveJammingAve3(j) = sum(SINRActiveJammingSet3) / (Count);
%     
%     SNRPowerWoIRSAve(j) = sum(SNRPowerWoIRSSet) / (Count);
end



% SNRPowerProAvedBm = 10 .* log10(SNRPowerProAve);
% 
% SINRPowerActiveJammingAvedBm1 = 10 .* log10(SINRPowerActiveJammingAve1);
% SINRPowerActiveJammingAvedBm2 = 10 .* log10(SINRPowerActiveJammingAve2);
% SINRPowerActiveJammingAvedBm3 = 10 .* log10(SINRPowerActiveJammingAve3);
% 
% SNRPowerWoIRSAvedBm = 10.* log10(SNRPowerWoIRSAve );
% 
% figure(1)
% plot(PdBmSet, SNRPowerProAvedBm, 'r-o', 'LineWidth',2);
% hold on
% plot(PdBmSet, SINRPowerActiveJammingAvedBm1, 'm-d', 'LineWidth',2);
% plot(PdBmSet, SINRPowerActiveJammingAvedBm2, 'c-p', 'LineWidth',2);
% plot(PdBmSet, SINRPowerActiveJammingAvedBm3, 'k-*', 'LineWidth',2);
% plot(PdBmSet, SNRPowerWoIRSAvedBm, '-s', 'LineWidth',2);
% set(gca,'xtick',[10,15,20,25,30,35,40]);
% xlabel('Transmit power of the LT (dBm)');
% ylabel('SNR/SINR (dB)');
% legend('Proposed scheme', 'Active jamming, P_a = 15 dBm', 'Active jamming, P_a = 25 dBm', 'Active jamming, P_a = 35 dBm',  'Without jamming');
% 
% save all













