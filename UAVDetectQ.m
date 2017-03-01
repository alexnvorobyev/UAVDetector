close all;
clear;
clc;

%% Загружаем данные 
load('AlexNetQuadcopter1.mat');
%% Пороговое значение яркости
threshold = 94;
se = strel('disk', 4);
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
hblob.MinimumBlobArea = 70;
hblob.MaximumBlobArea = 1100;
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
    if(frameNumber==280)
        stop=1;
    end
    
    videoFrame = videoFReader();
    videoFrameInfo  = videoFrame;
    
    image = rgb2gray(videoFrame);
    BW = image < threshold;
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
        x0(x0 < 1)=1;
        xn(xn > w)=w;
        y0(y0 < 1)=1;
        yn(yn > h)=h;
        
        %         label_str = cell(1,n);
        for i=1:1:n
            %             label_str{i} = ['S ', num2str(area(i),'%0.0f')];
            imageRio = videoFrame(y0(i):yn(i), x0(i):xn(i), :);
            image227 = im2uint8(imresize(imageRio, [227 227]));
            
            imageFeatures = activations(convnet, image227, featureLayer);
            label = predict(classifier, imageFeatures);
            
            if (label=='Quadcopter')
                
                % Для отладки
%                 if((frameNumber - frameDetect)>1)
%                     stop=1;
%                 end
%                 frameDetect = frameNumber;                
%                 figure(1)
%                 imshow(imageRio);
%                 title([cellstr(label)]);
%                 title([cellstr(label), num2str(frameNumber)]);
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

