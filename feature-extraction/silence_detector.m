%Feature Extraction for a sliding win
%configuration for phone on table : 9.65, 10

filename='humancall/robocom5.csv';
fnameString="test"; 
fnameNeumericStart=1; 

%class="human";
start=104;
savedStartPoint=start;


num = csvread(filename) ;
[r,c] = size(num) ;
timeValue=num(:,1); 
endVal=timeValue(end);
endVal=(endVal-10)*10;
disp(endVal);

%Delete unecessary information
num(2:2:end,:) = [] ;


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
Fs = 1/mean(diff(num(:,1)));  
y_highpass=highpass(num(:,4),18,Fs);
num(:,4)=y_highpass;


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
areaCount=0;
silentTime=0;
silentDifference=0;




for x=1:100
     iterate=num;
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
                                fprintf('Possible Error before that %d\n',appStart);

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
        disp(appStart);
        disp(appEnd);
        disp("==========");
        
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
    fprintf('appStart %.2f\n', appStart);
    
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


fprintf('Total word region found %d\n',regionCount);
fprintf('Total possible error found %d\n',errorCount);
fprintf('Total oversized word regions %d\n',largeCount);
fprintf('Total spiked percentage %d\n',areaCount);  
fprintf('Total Silent Time %.2f\n', silentTime);


%my_xls(filename,savedStart,savedEnd,endVal,class);








