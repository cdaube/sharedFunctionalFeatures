function handles = hpdi(data,varargin)

    % fetch optional arguments
    fixedArgs = 2;
    if nargin >= fixedArgs+1
        for ii = fixedArgs+1:2:nargin
            switch varargin{ii-fixedArgs}
                case 'offsets'
                    offsets = varargin{ii-(fixedArgs-1)};
                case 'pdInner'
                    pdInner = varargin{ii-(fixedArgs-1)};
                case 'pdOuter'
                    pdOuter = varargin{ii-(fixedArgs-1)};
                case 'widthInner'
                    widthInner = varargin{ii-(fixedArgs-1)};
                case 'widthOuter'
                    widthOuter = varargin{ii-(fixedArgs-1)};
                case 'cMap'
                    cMap = varargin{ii-(fixedArgs-1)};
            end
        end
    end
    
    % set defaults
    if ~exist('offsets','var'); offsets = 'MAP'; end
    if ~exist('widthInner','var'); widthInner = 5; end
    if ~exist('widthOuter','var'); widthOuter = 1; end
    if ~exist('pdInner','var'); pdInner = 50; end
    if ~exist('pdOuter','var'); pdOuter = 95; end
    if ~exist('cMap','var'); cMap = lines(size(data,3)); end
    
    invert = @(x) not(x-1)+1;
    
    handles = cell(size(data,2),size(data,3));
    
    for ii = 1:size(data,3)
        for dd = 1:size(data,2)

            landMarks = prctile(data(:,dd,ii),[(100-pdOuter)/2 (100-pdInner)/2 (100-pdInner)/2+pdInner (100-pdOuter)/2+pdOuter]);
            
            if strcmp(offsets,'MAP')
               [f,x] =  ksdensity(data(:,invert(dd),ii));
               [~,loc] = max(f);
               offsetLoc = x(loc);
            elseif strcmp(offsets,'mean')
                offsetLoc = mean(data(:,invert(dd),ii));
            elseif ~ischar(offsets)
                offsetLoc = offsets(dd);
            end
            
            if dd == 1
                handles{dd,ii} = plot([offsetLoc(dd,ii) offsetLoc],[landMarks(1) landMarks(4)],'Color',cMap(ii,:),'LineWidth',widthOuter);
                hold on
                handles{dd,ii} = plot([offsetLoc(dd,ii) offsetLoc],[landMarks(2) landMarks(3)],'Color',cMap(ii,:),'LineWidth',widthInner);
            elseif dd == 2
                handles{dd,ii} = plot([landMarks(1) landMarks(4)],[offsetLoc offsetLoc],'Color',cMap(ii,:),'LineWidth',widthOuter);
                hold on
                handles{dd,ii} = plot([landMarks(2) landMarks(3)],[offsetLoc offsetLoc],'Color',cMap(ii,:),'LineWidth',widthInner);
            end

        end
    end
    
    hold off