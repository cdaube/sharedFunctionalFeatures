  library(R.matlab)
  library(brms)
  library(rstan)
  
  # set root path corresponding to the machine the code is run on
  thsNodename <- substr(Sys.info()["nodename"], start = 1, stop = 3)
  if (thsNodename =="chr"){
    rootDir <- "/media/"
    library(sjPlot)
  }else if (thsNodename =="mat"){
    rootDir <- "/analyse/"
  }else if (thsNodename =="tia"){
    rootDir <- "/analyse/"}
  
  # set other folders
  sourceFolder <- "Project0257/humanReverseCorrelation/rTables/"
  destinFolder <- "Project0257/humanReverseCorrelation/rModels/"
  outputFolder <- "Project0257/humanReverseCorrelation/rOutput/"
  
  fileNameSource <- paste("ReconVsVerid_MAE.mat", sep = "")
  # load data, create data frame, name variables
  thsTable <- readMat(paste(rootDir,sourceFolder,fileNameSource,sep = ""))
  thsTable <- data.frame(Reduce(rbind, thsTable))
  names(thsTable) <- c("coll","pps","fspc","human","texture","shape","PCA",
                  "triplet","ClassID","ClassMulti","AE","viAE","r")
  

  #if(FALSE) {
  # brms
  brmsModel <- brm(r ~ 0 + (1|fspc:pps) + (1|coll:pps) + 
                   human + PCA + shape + texture + triplet + ClassID + ClassMulti + AE + viAE,
                   data = thsTable, family = gaussian(),
                   cores = 4, iter = 5000, chains = 4, warmup = 1000,
                   control = list(adapt_delta = .95, max_treedepth = 15))
  
  # save brms fit ...
  brmsFit <- brmsModel$fit
  extractedFit <- extract(brmsFit)
  writeMat(paste(rootDir,destinFolder,"extractedFit_ReconVSGTMAE.mat",sep = ""),extractedFit=extractedFit)
  # ... including names ...
  writeMat(paste(rootDir,destinFolder,"extractedFit_ReconVSGTMAE_names.mat",sep = ""),names=names(brmsFit))
  # ... and model
  save(brmsModel, file=paste(rootDir,destinFolder,"brmsModel_ReconVSGTMAE.rda",sep = ""))
  #}
  
  # hypotheses
  load(paste(rootDir,destinFolder,"brmsModel_ReconVSGTMAE.rda",sep = ""))
  H <- hypothesis(brmsModel, "b_AE - b_human > 0", class = NULL)
  H <- hypothesis(brmsModel, "b_AE - b_triplet > 0", class = NULL)
  H <- hypothesis(brmsModel, "b_ClassMulti - b_ClassID > 0", class = NULL)  
  colMeans(H$samples < 0)
  
  featureSpaces <- c("human","PCA","shape","texture","triplet","ClassID","ClassMulti","AE","viAE")

  nFspc <- length(featureSpaces)
  evidenceRatios <- matrix(data=0,nrow=nFspc,ncol=nFspc)
  pp <- matrix(data=0L,nrow=nFspc,ncol=nFspc)
  for (fspc1 in 1:nFspc){
    for (fspc2 in 1:nFspc){
      if (fspc1 == fspc2) next
      #H <- hypothesis(brmsModel, paste("r_fspc[",fspc1,",Intercept] - r_fspc[",fspc2,",Intercept] > 0",sep = ""), class = NULL)
      H <- hypothesis(brmsModel, paste("b_",featureSpaces[fspc1]," - b_", featureSpaces[fspc2]," > 0",sep = ""), class = NULL)
      evidenceRatios[fspc1,fspc2] <- H$hypothesis$Evid.Ratio
      pp[fspc1,fspc2] <- colMeans(H$samples > 0)
    }
  }
  writeMat(paste(rootDir,destinFolder,"hypotheses_ReconVSGTMAE.mat",sep = ""),evidenceRatios=evidenceRatios,pp=pp)
  