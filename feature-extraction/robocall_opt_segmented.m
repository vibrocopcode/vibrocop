%Feature Extraction for a sliding win
%configuration for phone on table : 9.65, 10
savedStartValue = [13, 44, 42, 22, 17, 20, 22, 19, 21, 20, 21, 21, 18, 15, 16, 22, 16, 8. 24, 14, 19, 18, 20, 21, 21, 19, 13, 22, 24, 24, 24, 19, 21, 19, 18, 33, 20, 23, 20, 23, 17, 19];
%[numbers, strings, raw] = xlsread('acctest.xls');
%global nextRow;
%nextRow = size(raw, 1);

for i = 1:42

    filename = ['humancall/op7TLoudRobo', num2str(i), '.csv'];
    fnameString="test"; 
    fnameNeumericStart=1; 

    class="robo";
    savedStart = savedStartValue(i);
    
    % Display the filename and startValue
    disp(['Processing file: ', filename, ' with startValue: ', num2str(savedStart)]);
    %savedEnd=savedStart+10;

    num = csvread(filename) ;
    [r,c] = size(num) ;
    timeValue=num(:,1); 
    endVal=timeValue(end);
    endVal=(endVal-10)*10;
    %disp(endVal);

    %Delete unecessary information
    num(2:2:end,:) = [] ;
    

    regionCount=0;
    consecutiveChecker=0;
    consecutiveStart=0;
    consecutiveEnd=0;


    zerocount=0;
    onecount=0;
    twocount=0;
    threecount=0;
    fourcount=0;
    fivecount=0;
    sixcount=0;
    sevencount=0;
    eightcount=0;
    ninecount=0;

    %Fs = 1/mean(diff(num(:,1)));  
    %y_highpass=highpass(num(:,4),20,Fs);
    %num(:,4)=y_highpass;
    
    %high pass filter
    Fs = 1/mean(diff(num(:,1)));  
    %y_highpass=highpass(num(:,4),30,Fs);
    %num(:,4)=y_highpass;


    calculate=num;

     %Delete rows for specific condition
      clowIndices = find(calculate(:,1)<10);
      calculate(clowIndices,:) = []; 

      chighIndices = find(calculate(:,1)>80);
      calculate(chighIndices,:) = [];

      mainZ=calculate(:,4) ;
      meanV=mean(mainZ);

      %disp(mainZ)



    notInside=0;
    startObserver=0;
    startConsecutive=0;
    appStart=0;
    endConsecutive=0;
    tempEnd=0;
    appEnd=0;
    endObserver=0;
    regionCount=0;
    errorCount=0;
    errorState=0;  
    largeCount=0;
    smallCount=0;
    %xlimit=10;
    
    % Define the array of i values where xlimit should be 8
    
    if (class == "robo")
        
        i_values = [45, 46];
    else
        i_values =[45, 46];
        
    end
    
    if ismember(i, i_values)
        xlimit = 8;
    else
        xlimit = 12;
    end
    

    for x=1:xlimit
        %disp(savedStart);

        savedEnd=savedStart+10;
        disp(x);
        disp("Enter");
        disp(savedEnd);
        my_xls(filename,savedStart,savedEnd,endVal,class,num,Fs,x);
        savedStart=savedStart+10;
    end
     % Display finished processing message
    disp(['Finished processing file: ', filename]);
end



function my_xls(filename,start,funcend,endVal,class,num,Fs,x)



startValue=start;
endValue=funcend;
%watchCompareStart=20;
%watchCompareEnd=34;
phoneCompareStart=10;
phoneCompareEnd=endVal+10;
%disp(watchStartValue);
%disp(endValue);
%disp(watchEndValue);

%now starting the smartphonw calculations

numPhone = csvread(filename) ;
[r1,c1] = size(numPhone) ;

numPhone(2:2:end,:) = [] ;



comparePhone=numPhone;

%Delete rows for phone compare values
comparePLow = find(comparePhone(:,1)<phoneCompareStart);
comparePhone(comparePLow,:) = [];

comparePHigh = find(comparePhone(:,1)>phoneCompareEnd);
comparePhone(comparePHigh,:) = [];

FsT= 1/mean(diff(numPhone(:,1)));  


%Delete rows for specific condition in Phone
lowIndicesp = find(numPhone(:,1)<startValue);
numPhone(lowIndicesp,:) = [];

highIndicesp = find(numPhone(:,1)>endValue);
numPhone(highIndicesp,:) = [];


analysePhone=numPhone;

Fsp = 1/mean(diff(numPhone(:,1)));  
Fn=Fsp/2;
y_highpass=highpass(numPhone(:,4),30,Fsp);
numPhone(:,4)=y_highpass;



phoneRms = numPhone(:,5) ;

phoneZ=numPhone(:,4);
comparePZ=comparePhone(:,4);

%silent time code

[silentTime,totalPeak]=silentTimeSelector(analysePhone,startValue);

%fprintf('Total spiked percentage %d\n',totalPeak);  
%fprintf('Total Silent Time %.2f\n', silentTime);

%Calculating Frequency domain features


Tr = linspace(numPhone(1,1), numPhone(1,end), size(numPhone,1));  
Dr = resample(phoneZ, Tr); 
Dr_mc  = Dr - mean(Dr,1); 


FDr_mc = fft(Dr_mc, [], 1);
Fv = linspace(0, 1, fix(size(FDr_mc,1)/2)+1)*Fn; 

Iv = 1:numel(Fv); 
amplitude=abs(FDr_mc(Iv,:))*2;

upperPart=Fv*amplitude;
ampSum=sum(amplitude);

specCentroid=upperPart/ampSum;
%disp(specCentroid); 

FvSqr=Fv.^2;
stdDevupper=FvSqr*amplitude;
specStdDev=sqrt(stdDevupper/ampSum);
specCrest=max(amplitude)/specCentroid;


specSkewness=(((Fv-specCentroid).^3)*amplitude)/(specStdDev)^3;

specKurt=(sum((((amplitude-specCentroid).^4).*amplitude))/(specStdDev)^4)-3 ;
maxFreq=max(Fv);
maxMagx=max(phoneZ);







meanP=mean(phoneZ);
minP=min(phoneZ);
maxP=max(phoneZ);
meanPZ=mean(comparePZ);
%gradientZ=mean(gradient(phoneZ));
%disp(meanPZ); 
irrk=irregularityk(phoneZ);
irrj=irregularityj(phoneZ);
sharp=sharpness(phoneZ);
smooth=smoothness(phoneZ);

%now adding frequency domain things:





%disp(meanP);
%disp(minP);
%disp(maxP);

meanCrossingP=phoneZ > meanPZ;
numberCrossingP=sum(meanCrossingP(:) == 1);
meanCrossingRateP=numberCrossingP/numel(phoneZ);
%disp(meanCrossingRateP);



%Extracting frequency domain values:

Fp = fft(phoneZ,1024);
FFTCoEffp=Fp/length(phoneZ);
powp = Fp.*conj(Fp);
total_powp = sum(powp);
%disp(total_powp);



Fsp = 1/mean(diff(numPhone(:,1)));

penp=pentropy(phoneZ,Fsp);
sumPenp=sum(penp);

%disp(sumPenp);

%centroid=spectralCentroid(phoneZ,Fsp);
%disp(centroid)

%sharpness = acousticSharpness(phoneZ,Fsp);
%disp(sharpness);



hdp = dfilt.fftfir(phoneZ,1024);
cp=fftcoeffs(hdp);
ampp = 2*abs(cp)/length(phoneZ);
phasep=angle(cp);
magnitudep=abs(ampp);

highestMagp=max(magnitudep);
sumMagp=sum(magnitudep);

frequency_ratiop=highestMagp/sumMagp;


%now the signal shape features

% Number of peaks
numPeaks = length(findpeaks(phoneZ));

% Zero-crossing rate
zeroCrossingRate = length(find(phoneZ(1:end-1).*phoneZ(2:end) < 0));

% Slope changes
slopeChanges = length(find(diff(phoneZ(2:end)) .* diff(phoneZ(1:end-1)) < 0));

% Number of inflection points
numInflectionPoints = length(find(diff(diff(phoneZ)) > 0));






% Assuming 'signal' is your data vector
signalLength = length(phoneZ);

% Calculate the maximum decomposition level
maxDecompLevel = fix(log2(signalLength));

% Display the maximum decomposition level
%fprintf('The maximum decomposition level for the given signal is %d.\n', maxDecompLevel);

% Choose a wavelet and level of decomposition
waveletFunction = 'db4';
decompositionLevel = 10;

% Perform the Discrete Wavelet Transform
[C, L] = wavedec(phoneZ, decompositionLevel, waveletFunction);

% Extract the approximation and detail coefficients
approximationCoefficients = appcoef(C, L, waveletFunction);
detailCoefficients = detcoef(C, L, 'cells');

% Save the first 10 detail coefficients into separate variables
for i = 1:10
    eval(sprintf('d%d = detailCoefficients{%d};', i, i));
    eval(sprintf('total_d%d = sum(d%d);', i, i)); % Calculate the total coefficient value 
end

%disp(total_d10);




statX=[mean(phoneZ) max(phoneZ) min(phoneZ) std(phoneZ) var(phoneZ) range(phoneZ) (std(phoneZ)/mean(phoneZ))*100 skewness(phoneZ) kurtosis(phoneZ) quantile(phoneZ,[0.25,0.50,0.75]) meanCrossingRateP total_powp sumPenp frequency_ratiop irrk irrj sharp smooth specCentroid specStdDev specCrest specSkewness specKurt maxFreq maxMagx numPeaks zeroCrossingRate slopeChanges numInflectionPoints totalPeak silentTime total_d1 total_d2 total_d3 total_d4 total_d5 total_d6 total_d7 total_d8 total_d9 total_d10 class];


%disp(mean(phoneZ));
t=array2table(statX);

%global nextRow;
%nextRow = nextRow + 1;
%disp(nextRow);
%cellReference = sprintf('A%d', nextRow);
%xlswrite('acctest.xls', statX, 'Sheet1', cellReference);



%disp(d1);
%disp(d2);



%[numbers, strings, raw] = xlsread('acctest.xls'); % Read the numeric data from the Excel file
%[numRows, ~] = size(numbers); % Get the number of rows in the numeric data

%strings = {}; % Create an empty cell array to store the string data

% Perform your calculations and generate the output
% Assuming you have the output values in a variable called "outputValues"

% Determine the next available row in the Excel file
%[~, ~, existingData] = xlsread('acctest.xls', 'Sheet1');
%nextRow = size(existingData, 1) + 1;
%disp(nextRow);
global nextRow;
nextRow=nextRow+1;

% Write the outputValues to the appropriate columns in the Excel file
cellReference = sprintf('A%d', nextRow);
%xlswrite('acctest.xls', statX, 'Sheet1', cellReference);
% Convert statX to a row matrix

% Convert statX to a row cell array
statX_cell = num2cell(statX);

% Specify the file name

outputfileHeader = "op7TLoud";
outputFileNumber = x;
fileExtension=".csv";


% Concatenate the strings
fullFileName = outputfileHeader + num2str(outputFileNumber) + fileExtension;

disp(fullFileName);

outfilename = fullFileName;

% Append statX to the CSV file
if exist(outfilename, 'file')
    % If the file already exists, append to it
    writetable(cell2table(statX_cell), outfilename, 'WriteVariableNames', false, 'WriteMode', 'append');
else
    % If the file doesn't exist, create a new one
    writetable(cell2table(statX_cell), outfilename, 'WriteVariableNames', false);
end


end

function [silentTime,totalPeak]=silentTimeSelector(analysePhone,startValue)
    start=startValue;
    savedStartPoint=start;
    
    
    
    regionCount=0;
    consecutiveChecker=0;
    consecutiveStart=0;
    consecutiveEnd=0;
    savedEnd=0;



    zerocount=0;
    onecount=0;
    twocount=0;
    threecount=0;
    fourcount=0;
    fivecount=0;
    sixcount=0;
    sevencount=0;
    eightcount=0;
    ninecount=0;

    %Fs = 1/mean(diff(num(:,1)));  
    %y_highpass=highpass(num(:,4),20,Fs);
    %num(:,4)=y_highpass;
    %high pass filter
    Fs = 1/mean(diff(analysePhone(:,1)));  
    y_highpass=highpass(analysePhone(:,4),18,Fs);
    analysePhone(:,4)=y_highpass;
    
    
    
     calculate=analysePhone;

     %Delete rows for specific condition
      clowIndices = find(calculate(:,1)<10);
      calculate(clowIndices,:) = []; 

      chighIndices = find(calculate(:,1)>80);
      calculate(chighIndices,:) = [];

      mainZ=calculate(:,4) ;
      meanV=mean(mainZ);

      %disp(mainZ)



        notInside=0;
        startObserver=0;
        startConsecutive=0;
        appStart=0;
        endConsecutive=0;
        tempEnd=0;
        appEnd=0;
        endObserver=0;
        regionCount=0;
        errorCount=0;
        errorState=0;  
        largeCount=0;
        smallCount=0;
        areaCount=0;
        silentTime=0;
        silentDifference=0;
        
        
        
       for x=1:100
             iterate=analysePhone;
            %disp(iterate);
            start=start+0.1;
            initValue=start;
            final=start+0.1;

            %Delete rows for specific condition
            lowIndices = find(iterate(:,1)<initValue);
            iterate(lowIndices,:) = [];

            highIndices = find(iterate(:,1)>final);
             iterate(highIndices,:) = [];

            %disp(lowIndices);





            %Extract all axes
            ax = iterate(:,2) ;
            ay = iterate(:,3) ;
            az = iterate(:,4) ;
            %disp(az);

            meanX=mean(az);
            %disp(meanX);
            minX=min(az);
            maxX=max(az);

           % disp(start);
           % disp(min(az));
            %disp(max(az));
             try
                 if min(az)<=-0.0025 || max(az)>=0.0025
                        %disp(start);
                        areaCount=areaCount+1;
                        if(notInside==0 && startObserver==0)
                            startObserver=1;
                            tempStart=start;
                            startConsecutive=1;        
                        elseif(notInside==0 && startObserver==1)
                                startConsecutive=startConsecutive+1;
                                if(startConsecutive>=2)
                                    notInside=1;
                                    startObserver=0;
                                    appStart=tempStart;
                                    %fprintf('The starting Point is %d\n',appStart);
                                    if(appStart-savedEnd)>0.3 && (appStart-errorState)>1.0
                                        errorCount=errorCount+1;
                                        %disp(appStart);
                                        errorState=appStart;
                                        %fprintf('Possible Error before that %d\n',appStart);

                                    end
                                end
                        elseif(notInside==1 && endObserver==1)
                            endObserver=0;
                            tempEnd=0;
                        end


                 else
                        if (notInside==1 && endObserver==0)
                            endConsecutive=1;
                            endObserver=1;
                            tempEnd=final;
                            %disp(tempEnd);
                            %disp(tempEnd-appStart);
                            if(tempEnd-appStart>0.3) 
                               %disp(tempEnd);
                               appEnd=tempEnd;
                                notInside=0; 
                            end

                        elseif (notInside==1 && endObserver==1)
                            endConsecutive=endConsecutive+1;
                            if(endConsecutive>=1)
                                if(tempEnd-appStart>0.3)
                                    appEnd=tempEnd;
                                    notInside=0;
                                end

                            end
                            if(endConsecutive>=2)
                                appEnd=tempEnd;
                                notInside=0;              
                            end

                        else
                            if(notInside==0 && startObserver==1)
                                startObserver=0;
                                tempStart=0;
                            end
                        end  

                    end 

                catch
                    disp("Finished");
                    break
             end


             if(appStart~=0 && appEnd~=0 && appStart<appEnd && appEnd-appStart>=0.8 && appEnd-appStart<=8.0)
                %disp(savedEnd);
                if (savedEnd==0)
                    silentDifference=appStart-savedStartPoint;
                else
                    silentDifference=appStart-savedEnd;
                end

                %disp(silentDifference);

                silentTime=silentTime+silentDifference;
                %disp(appStart);
                %disp(appEnd);
                %disp("==========");

                if(appEnd-appStart)>=0.8
                    largeCount=largeCount+1;
                elseif(appEnd-appStart)<=0.3
                    smallCount=smallCount+1;
                end
                regionCount=regionCount+1;
                savedStart=appStart;
                savedEnd=appEnd;
                appStart=0;
                appEnd=0;
             end
       end
        
       
       
        if(appStart>appEnd)
            %fprintf('appStart %.2f\n', appStart);

            if (savedEnd==0)
                    silentDifference=appStart-savedStartPoint;
                else
                    silentDifference=appStart-savedEnd;
            end
            silentTime=silentTime+silentDifference;
        else


            silentDifference= (savedStartPoint+10)-savedEnd;
            silentTime=silentTime+silentDifference;

            if(silentTime>10)
                silentTime=10;
            end

         end

    totalPeak=areaCount;
end


function ikSum=irregularityk(phonez)
%disp(phonez);
N=[10 100 1000];

ikSum=0;
for i=1:length(phonez)-2
   ik=(phonez(i+1)-(phonez(i)+phonez(i+1)+phonez(i+2)/3));
   ikSum=ikSum+ik;
    
end
end

function ijSum=irregularityj(phonez)


ijSum=0;
for i=1:length(phonez)-1
   ij1=(phonez(i)-phonez(i+1))^2;
   ij2=phonez(i)^2;
   ij=ij1/ij2;
   ijSum=ijSum+ij;
    
end
end

function finalsharp=sharpness(phonez)
sharpn=0;
tempi=0;
for i=1:length(phonez)
    if(i<15)
        tempi=real(i*phonez(i)^0.23);
        %disp(tempi);
    else
        tempi=real(0.066*exp(0.171*i)*i*phonez(i)^0.23);
    end
    
    sharpn=sharpn+tempi;
end

finalsharp=(0.11*sharpn)/length(phonez);
end

function smoothSum=smoothness(phonez)

smoothSum=0;
for i=1:length(phonez)-2
   
    ismooth=real((20*log(phonez(i))-(20*log(phonez(i))+20*log(phonez(i+1))+20*log(phonez(i+2))))/3);
    
    smoothSum=smoothSum+ismooth;
end


end








