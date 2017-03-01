close all;
clear;
clc;

%% Загружаем данные
load('AlexNetUAVQ.mat');
%% Пороговое значение яркости
threshold = 87; %94;
se = strel('disk', 5);
%% Указываем видеофайл
[FileName,PathName] = uigetfile('*.*','Select Input Video File');
videoFReader = vision.VideoFileReader([PathName,FileName],'VideoOutputDataType', 'uint8');

 VideoFWriter = VideoWriter('VideoOut.avi');
 VideoFWriter.FrameRate = 25;

%% Создаем объект для отображения видео
videoPlayer = vision.VideoPlayer;

%% Объект для поиска Блобов на изображении
hblob = vision.BlobAnalysis;
hblob.AreaOutputPort = true;
hblob.CentroidOutputPort = true;
hblob.ExcludeBorderBlobs = true;
hblob.MinimumBlobArea = 87; %82;
hblob.MaximumBlobArea = 1250;
%% Переменные
frameNumber = 0;
frameDetect = 0;
bbox_rio = 50;

videoFrame = videoFReader();
[h,w,d]=size(videoFrame);

 open(VideoFWriter)
while ~isDone(videoFReader)
    
    frameNumber = frameNumber + 1;
    % Для отладки
    if(frameNumber>=280)
        stop=1;
    end
    
    videoFrame = videoFReader();
    videoFrameInfo  = videoFrame;
    image = rgb2gray(videoFrame);
    BW = image < threshold;
    
%     BW=bwpropfilt(BW,'Area',[3, 2000]);
    BW2 = imdilate(BW,se);
    [area, centroid, bboxes] = hblob(BW2);

    n=length(area);
    if n>0
        
        % Пересчет координат объектов
        bboxCenter = round(centroid);
        x0 = bboxCenter(:,1)- bbox_rio;
        xn = bboxCenter(:,1)+ bbox_rio;
        y0 = bboxCenter(:,2)- bbox_rio;
        yn = bboxCenter(:,2)+ bbox_rio;
        
        % Граничные условия
        x0_z=(x0 < 1);
        x0(x0_z)=1;
        xn(x0_z)=2*bbox_rio;
        
        xn_m=(xn > w);
        xn(xn_m)=w;
        x0(xn_m)=w-2*bbox_rio;
        
        y0_z=(y0 < 1);
        y0(y0_z)=1;
        yn(y0_z)=2*bbox_rio;;
        
        yn_m=(yn > h);
        yn(yn_m)=h;
        y0(yn_m)=h-2*bbox_rio;

        %         label_str = cell(1,n);
        for i=1:1:n
            %             label_str{i} = ['S ', num2str(area(i),'%0.0f')];
            imageRio = videoFrame(y0(i):yn(i), x0(i):xn(i), :);
            image227 = im2uint8(imresize(imageRio, [227 227]));
            
            imageFeatures = activations(convnet, image227, featureLayer);
            label = predict(classifier, imageFeatures);
%             imshow(imageRio);
%             title([cellstr(label)]);
            
            if (label=='UAV')
                
                % Для отладки
                %                 if((frameNumber - frameDetect)>1)
                %                     stop=1;
                %                 end
                %                 frameDetect = frameNumber;
                                imshow(imageRio);
                                title([cellstr(label)]);
                videoFrameInfo = insertObjectAnnotation(videoFrame,'Rectangle',bboxes(i,:),['UAV S:',num2str(area(i),'%0.0f')]...
                    , 'Color', 'g','TextBoxOpacity',0.3,'FontSize',10);
            end
        end %for
        %         videoFrameInfo = insertObjectAnnotation(videoFrame, 'rectangle', bboxes, label_str, 'Color', 'g','TextBoxOpacity',0.4,'FontSize',10);
        
    end %if(n>0)
    
    videoPlayer(videoFrameInfo);
         writeVideo(VideoFWriter,videoFrameInfo)
end

 close(VideoFWriter)
release(videoPlayer);
release(videoFReader);

