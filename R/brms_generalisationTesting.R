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
    rootDir <- "/analyse/"
  }else if (thsNodename =="dee"){
    rootDir <- "/analyse/"}
  
  # set other folders
  sourceFolder <- "Project0257/humanReverseCorrelation/rTables/"
  destinFolder <- "Project0257/humanReverseCorrelation/rModels/"
  outputFolder <- "Project0257/humanReverseCorrelation/rOutput/"
  
# run task-wise models for all systems (otherwise, number of data points infeasible with MCMC)  
if (FALSE) {
for (tt in 5){
    print(paste("task",tt))

    #nSkip <- 8

    fileNameSource <- paste("generalisationTestingErrT",tt,".mat", sep = "")
    thsTable <- readMat(paste(rootDir,sourceFolder,fileNameSource,sep = ""))
    thsTable <- data.frame(Reduce(rbind, thsTable))
    names(thsTable) <- c("task","left","middle","right","age","sex","amp","diag","sys",
                      "texture","shape","PCA","triplet","ClassID","ClassMulti","AE","viAE","VAE",
                      "texture_euc","shape_euc","PCA_euc","triplet_euc","ClassID_euc","ClassMulti_euc","AE_euc","viAE_euc","VAE_euc",
                      "texture_eucFit","shape_eucFit","PCA_eucFit","triplet_eucFit","ClassID_eucFit",
                      "ClassMulti_eucFit","AE_eucFit","viAE_eucFit","VAE_eucFit",
                      "VAE2","VAE5","VAE10","VAE20",
                      "ClassID_dn","ClassMulti_dn","VAE_ldn","VAE_nldn",
                      "coll","val","pps","err")

    #newTable <- thsTable[which(thsTable$sys>nSkip),]

    #if (FALSE) {
    brmsModel <- brm(err ~ 1 + (1|sys) + (1|coll:pps:val),
                  data = thsTable, family=cumulative("probit"),
                  cores = 4, iter = 5000, chains = 4, warmup = 1000,
                  control = list(adapt_delta = .95, max_treedepth = 15))
    #}

    # load model (if not fit in this session)
    # load(file=paste(rootDir,destinFolder,"brmsModel_generalisationTestingErr_T",tt,".rda",sep = ""))

    print(paste("printing chains check figure, task",tt))
    # save plot of chain traces and densities
    png(file=paste("MCumProb_collValPps_tt",tt,"_chains.png"))
    plot(brmsModel, N = 15, ask = FALSE)
    dev.off()

    print(paste("printing pp check figure, task",tt))
    # save plot of posterior predictive
    png(file=paste("MCumProb_collValPps_T",tt,"_ppCheck.png", sep = ""))
    pp_check(brmsModel)
    dev.off()

    print(paste("saving fit and model, task",tt))
    # save brms fit ...
    brmsFit <- brmsModel$fit
    extractedFit <- extract(brmsFit)
    writeMat(paste(rootDir,destinFolder,"extractedFit_generalisationTestingErr_T",tt,".mat",sep = ""),extractedFit=extractedFit)
    # ... including names ...
    writeMat(paste(rootDir,destinFolder,"extractedFit_generalisationTestingErr_T",tt,"_names.mat",sep = ""),names=names(brmsFit))
    # ... and model
    save(brmsModel, file=paste(rootDir,destinFolder,"brmsModel_generalisationTestingErr_T",tt,".rda",sep = ""))


    print(paste("extracting samples in favour of hypotheses, task",tt))
    # count samples in favour of hypotheses
    systems <- c("texture","shape","PCA","triplet","ClassID","ClassMulti","AE","viAE","VAE",
                 "texture_euc","shape_euc","PCA_euc","triplet_euc","ClassID_euc","ClassMulti_euc","AE_euc","viAE_euc","VAE_euc",
                 "texture_eucFit","shape_eucFit","PCA_eucFit","triplet_eucFit","ClassID_eucFit",
                 "ClassMulti_eucFit","AE_eucFit","viAE_eucFit","VAE_eucFit",
                 "VAE2","VAE5","VAE10","VAE20","ClassID_dn","ClassMulti_dn","VAE_ldn","VAE_nldn")

    nSys <- length(systems)

    evidenceRatios <- matrix(data=0,nrow=nSys,ncol=nSys)
    pp <- matrix(data=0L,nrow=nSys,ncol=nSys)
    for (sys1 in 1:nSys){
      for (sys2 in 1:nSys){
        if (sys1 == sys2) next

        H <- hypothesis(brmsModel, paste("r_sys[",sys1,",Intercept] - r_sys[",sys2,",Intercept] > 0",sep = ""), class = NULL)
        evidenceRatios[sys1,sys2] <- H$hypothesis$Evid.Ratio
        pp[sys1,sys2] <- colMeans(H$samples > 0)
      }
    }
    writeMat(paste(rootDir,destinFolder,"hypotheses_generalisationTestingErr_taskwise_T",tt,".mat",sep = ""),
        evidenceRatios=evidenceRatios,pp=pp)

    print(paste("finished task",tt))
}

}

# run model with all tasks for 8 main systems 
if (FALSE) {
  print(paste("run model with all tasks for 8 main systems"))
  fileNameSource <- paste("generalisationTestingErr.mat", sep = "")
  thsTable <- readMat(paste(rootDir,sourceFolder,fileNameSource,sep = ""))
  thsTable <- data.frame(Reduce(rbind, thsTable))
  names(thsTable) <- c("task","left","middle","right","age","sex","amp","diag","sys",
                       "texture","shape","PCA","triplet","ClassID","ClassMulti","AE","viAE","VAE",
                       "texture_euc","shape_euc","PCA_euc","triplet_euc","ClassID_euc","ClassMulti_euc","AE_euc","viAE_euc","VAE_euc",
                       "texture_eucFit","shape_eucFit","PCA_eucFit","triplet_eucFit","ClassID_eucFit",
                       "ClassMulti_eucFit","AE_eucFit","viAE_eucFit","VAE_eucFit",
                       "VAE2","VAE5","VAE10","VAE20",
                       "ClassID_dn","ClassMulti_dn","VAE_ldn","VAE_nldn",
                       "coll","val","pps","err")

  # subset to index only rows of main models 
  newTable <- thsTable[which(thsTable$sys<9),]

  #lme4Model <- lmer(err ~ 1 + (1|sys:task) + (1|pps:val:coll),data = thsTable)


  brmsModel <- brm(err ~ 1 + (1|sys:task) + (1|pps:val:coll),
                   data = newTable, family=cumulative("probit"),
                   cores = 4, iter = 5000, chains = 4, warmup = 1000,
                   control = list(adapt_delta = .95, max_treedepth = 15))

  
  png(file="MCumProb_5T_v5_collValPps_chains.png")
  plot(brmsModel, ask = FALSE, N = 15)
  dev.off()

  png(file="MCumProb_5T_v5_collValPps_ppCheck.png")
  pp_check(brmsModel)
  dev.off()

  # save brms fit ...
  brmsFit <- brmsModel$fit
  extractedFit <- extract(brmsFit)
  writeMat(paste(rootDir,destinFolder,"extractedFit_generalisationTestingErr5T_v5.mat",sep = ""),extractedFit=extractedFit)
  # ... including names ...
  writeMat(paste(rootDir,destinFolder,"extractedFit_generalisationTestingErr5T_v5_names.mat",sep = ""),names=names(brmsFit))
  # ... and model
  save(brmsModel, file=paste(rootDir,destinFolder,"brmsModel_generalisationTestingErr5T_v5.rda",sep = ""))



  
systems <- c("texture","shape","PCA","triplet","ClassID","ClassMulti","AE","viAE")

nSys <- length(systems)
for (tt in 1:5){
evidenceRatios <- matrix(data=0,nrow=nSys,ncol=nSys)
pp <- matrix(data=0L,nrow=nSys,ncol=nSys)
for (sys1 in 1:nSys){
  for (sys2 in 1:nSys){
    if (sys1 == sys2) next

    H <- hypothesis(brmsModel, paste("r_sys:task[",sys1,"_",tt,",Intercept] - r_sys:task[",
                                                   sys2,"_",tt,",Intercept] > 0",sep = ""), class = NULL)
    evidenceRatios[sys1,sys2] <- H$hypothesis$Evid.Ratio
    pp[sys1,sys2] <- colMeans(H$samples > 0)
  }
}
writeMat(paste(rootDir,destinFolder,"hypotheses_generalisationTestingErr5T_tt",tt,"_v5.mat",sep = ""),
    evidenceRatios=evidenceRatios,pp=pp)
}

}


# try to run model with all tasks for for all systems 
#if (FALSE) {
  print(paste("try run model with all tasks for all main systems"))
  fileNameSource <- paste("generalisationTestingErr.mat", sep = "")
  thsTable <- readMat(paste(rootDir,sourceFolder,fileNameSource,sep = ""))
  thsTable <- data.frame(Reduce(rbind, thsTable))
  names(thsTable) <- c("task","left","middle","right","age","sex","amp","diag","sys",
                       "texture","shape","PCA","triplet","ClassID","ClassMulti","AE","viAE","VAE",
                       "texture_euc","shape_euc","PCA_euc","triplet_euc","ClassID_euc","ClassMulti_euc","AE_euc","viAE_euc","VAE_euc",
                       "texture_eucFit","shape_eucFit","PCA_eucFit","triplet_eucFit","ClassID_eucFit",
                       "ClassMulti_eucFit","AE_eucFit","viAE_eucFit","VAE_eucFit",
                       "VAE2","VAE5","VAE10","VAE20",
                       "ClassID_dn","ClassMulti_dn","VAE_ldn","VAE_nldn",
                       "coll","val","pps","err")


  brmsModel <- brm(err ~ 1 + (1|sys:task) + (1|pps:val:coll),
                   data = thsTable, family=cumulative("probit"),
                   silent = 0, cores = 4, iter = 5000, chains = 4, warmup = 1000,
                   control = list(adapt_delta = .95, max_treedepth = 15))

  
  png(file="MCumProb_5TAllSys_v5_collValPps_chains.png")
  plot(brmsModel, ask = FALSE, N = 15)
  dev.off()

  png(file="MCumProb_5TAllSys_v5_collValPps_ppCheck.png")
  pp_check(brmsModel)
  dev.off()

  # save brms fit ...
  brmsFit <- brmsModel$fit
  extractedFit <- extract(brmsFit)
  writeMat(paste(rootDir,destinFolder,"extractedFit_generalisationTestingErr5TAllSys_v5.mat",sep = ""),extractedFit=extractedFit)
  # ... including names ...
  writeMat(paste(rootDir,destinFolder,"extractedFit_generalisationTestingErr5TAllSys_v5_names.mat",sep = ""),names=names(brmsFit))
  # ... and model
  save(brmsModel, file=paste(rootDir,destinFolder,"brmsModel_generalisationTestingErr5TAllSys_v5.rda",sep = ""))



  
systems <- c("texture","shape","PCA","triplet","ClassID","ClassMulti","AE","viAE","VAE",
             "texture_euc","shape_euc","PCA_euc","triplet_euc","ClassID_euc","ClassMulti_euc","AE_euc","viAE_euc","VAE_euc",
             "texture_eucFit","shape_eucFit","PCA_eucFit","triplet_eucFit","ClassID_eucFit",
             "ClassMulti_eucFit","AE_eucFit","viAE_eucFit","VAE_eucFit",
             "VAE2","VAE5","VAE10","VAE20","ClassID_dn","ClassMulti_dn","VAE_ldn","VAE_nldn")

nSys <- length(systems)
for (tt in 1:5){
evidenceRatios <- matrix(data=0,nrow=nSys,ncol=nSys)
pp <- matrix(data=0L,nrow=nSys,ncol=nSys)
for (sys1 in 1:nSys){
  for (sys2 in 1:nSys){
    if (sys1 == sys2) next

    H <- hypothesis(brmsModel, paste("r_sys:task[",sys1,"_",tt,",Intercept] - r_sys:task[",
                                                   sys2,"_",tt,",Intercept] > 0",sep = ""), class = NULL)
    evidenceRatios[sys1,sys2] <- H$hypothesis$Evid.Ratio
    pp[sys1,sys2] <- colMeans(H$samples > 0)
  }
}
writeMat(paste(rootDir,destinFolder,"hypotheses_generalisationTestingErr5TAllSys_tt",tt,"_v5.mat",sep = ""),
    evidenceRatios=evidenceRatios,pp=pp)
}

#}


# run model on error averaged across participant-individualised models
if (FALSE) {
  fileNameSource <- paste("generalisationTestingErrPAv.mat", sep = "")
  thsTable <- readMat(paste(rootDir,sourceFolder,fileNameSource,sep = ""))
  thsTable <- data.frame(Reduce(rbind, thsTable))
  names(thsTable) <- c("task","left","middle","right","age","sex","amp","diag","sys",
                       "texture","shape","PCA","triplet","ClassID","ClassMulti","AE","viAE","VAE",
                       "texture_euc","shape_euc","PCA_euc","triplet_euc","ClassID_euc","ClassMulti_euc","AE_euc","viAE_euc","VAE_euc",
                       "texture_eucFit","shape_eucFit","PCA_eucFit","triplet_eucFit","ClassID_eucFit",
                       "ClassMulti_eucFit","AE_eucFit","viAE_eucFit","VAE_eucFit",
                       "VAE2","VAE5","VAE10","VAE20",
                       "ClassID_dn","ClassMulti_dn","VAE_ldn","VAE_nldn",
                       "coll","val","pps","err")

  brmsModel <- brm(err ~ 1 + (1|sys:task) + (1|val:coll),
                   data = thsTable, family=gaussian,
                   cores = 4, iter = 5000, chains = 4, warmup = 1000,
                   control = list(adapt_delta = .95, max_treedepth = 15))

  
  png(file="MGauss_5TPAv_collValPps_chains.png")
  plot(brmsModel, ask = FALSE, N = 15)
  dev.off()

  png(file="MGauss_5TPAv_collValPps_ppCheck.png")
  pp_check(brmsModel)
  dev.off()

  # save brms fit ...
  brmsFit <- brmsModel$fit
  extractedFit <- extract(brmsFit)
  writeMat(paste(rootDir,destinFolder,"extractedFit_generalisationTestingErr5TPAv.mat",sep = ""),extractedFit=extractedFit)
  # ... including names ...
  writeMat(paste(rootDir,destinFolder,"extractedFit_generalisationTestingErr5TPAv_names.mat",sep = ""),names=names(brmsFit))
  # ... and model
  save(brmsModel, file=paste(rootDir,destinFolder,"brmsModel_generalisationTestingErr5TPAv.rda",sep = ""))



  
systems <- c("texture","shape","PCA","triplet","ClassID","ClassMulti","AE","viAE")

nSys <- length(systems)
for (tt in 1:5){
evidenceRatios <- matrix(data=0,nrow=nSys,ncol=nSys)
pp <- matrix(data=0L,nrow=nSys,ncol=nSys)
for (sys1 in 1:nSys){
  for (sys2 in 1:nSys){
    if (sys1 == sys2) next

    H <- hypothesis(brmsModel, paste("r_sys:task[",sys1,"_",tt,",Intercept] - r_sys:task[",
                                                   sys2,"_",tt,",Intercept] > 0",sep = ""), class = NULL)
    evidenceRatios[sys1,sys2] <- H$hypothesis$Evid.Ratio
    pp[sys1,sys2] <- colMeans(H$samples > 0)
  }
}
writeMat(paste(rootDir,destinFolder,"hypotheses_generalisationTestingErr5TPAv_tt",tt,".mat",sep = ""),
    evidenceRatios=evidenceRatios,pp=pp)
}

}
