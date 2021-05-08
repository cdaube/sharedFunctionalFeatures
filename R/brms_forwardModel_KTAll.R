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
  }else if (thsNodename =="dee"){
    rootDir <- "/analyse/"
  }else if (thsNodename =="tia"){
    rootDir <- "/analyse/"}
  
  # set other folders
  sourceFolder <- "Project0257/humanReverseCorrelation/rTables/"
  destinFolder <- "Project0257/humanReverseCorrelation/rModels/"
  outputFolder <- "Project0257/humanReverseCorrelation/rOutput/"
  
  fileNameSource <- paste("forwardModel_testKTAll.mat", sep = "")
  # load data, create data frame, name variables
  thsTable <- readMat(paste(rootDir,sourceFolder,fileNameSource,sep = ""))
  thsTable <- data.frame(Reduce(rbind, thsTable))
  names(thsTable) <- c("fold","coll","pps","fspc",
                       "Texture","Shape","pixelPCA","ShapeANDTexture","ShapeANDpixelPCA", 
                       "Triplet_emb","ClassID_emb","ClassMulti_emb","AE_emb","viAE_emb","VAE_emb","beta10VAE_emb", 
                       "ShapeANDAE_emb","ShapeANDviAE_emb","ShapeANDbeta1VAE_emb","ShapeANDClassMulti_embANDbeta1VAE",
                       "ShapeANDTextureANDAE_emb","ShapeANDTextureANDviAE_emb", 
                       "Texture_delta","Shape_delta","pixelPCA_delta","Triplet_delta","ClassID_delta",
                       "ClassMulti_delta","AE_delta","viAE_delta","beta1VAE_delta",
                       "ShapeVertex_delta","TexturePixel_delta", 
                       "Texture_deltalincomb","Shape_deltalincomb","pixelPCA_deltalincomb","Triplet_deltalincomb", 
                       "ClassID_deltalincomb","ClassMulti_deltalincomb","AE_deltalincomb", 
                       "viAE_deltalincomb","beta1VAE_deltalincomb", 
                       "ClassID_dn","ClassMulti_dn","VAE_ldn","VAE_nldn","KT")
                      
  
  # brms
  brmsModel <- brm(KT ~ 0 + (1|fspc:pps) + (1|fold:pps) + (1|coll:pps) +
                     Texture + Shape + pixelPCA + ShapeANDTexture + ShapeANDpixelPCA +
                     Triplet_emb + ClassID_emb + ClassMulti_emb + AE_emb + viAE_emb + VAE_emb + beta10VAE_emb +
                     ShapeANDAE_emb + ShapeANDviAE_emb + ShapeANDbeta1VAE_emb + ShapeANDClassMulti_embANDbeta1VAE +
                     ShapeANDTextureANDAE_emb + ShapeANDTextureANDviAE_emb +
                     Texture_delta + Shape_delta + pixelPCA_delta + Triplet_delta + ClassID_delta +
                     ClassMulti_delta + AE_delta + viAE_delta + beta1VAE_delta +
                     ShapeVertex_delta + TexturePixel_delta +
                     Texture_deltalincomb + Shape_deltalincomb + pixelPCA_deltalincomb + Triplet_deltalincomb +
                     ClassID_deltalincomb + ClassMulti_deltalincomb + AE_deltalincomb +
                     viAE_deltalincomb + beta1VAE_deltalincomb +
                     ClassID_dn + ClassMulti_dn + VAE_ldn + VAE_nldn,
                   data = thsTable, family = gaussian(),
                   cores = 4, iter = 5000, chains = 4, warmup = 1000,
                   control = list(adapt_delta = .95, max_treedepth = 15))
  
  png(file="M_forwardModelKT_chains.png",width=500,height=10000)
  plot(brmsModel, ask = FALSE, N = 50) # thank you Jack E. Taylor for making me aware of the N argument <3
  dev.off()

  png(file="M_forwardModelKT_ppCheck.png")
  pp_check(brmsModel)
  dev.off()


  # save brms fit ...
  brmsFit <- brmsModel$fit
  extractedFit <- extract(brmsFit)
  writeMat(paste(rootDir,destinFolder,"extractedFit_forwardModelKTAll.mat",sep = ""),extractedFit=extractedFit)
  # ... including names ...
  writeMat(paste(rootDir,destinFolder,"extractedFit_forwardModelKTAll_names.mat",sep = ""),names=names(brmsFit))
  # ... and model
  save(brmsModel, file=paste(rootDir,destinFolder,"brmsModel_forwardModelKTAll.rda",sep = ""))
  
  # hypotheses
  load(paste(rootDir,destinFolder,"brmsModel_forwardModelKTAll.rda",sep = ""))
  H <- hypothesis(brmsModel, "b_viAE_emb - b_ClassMulti_emb > 0", class = NULL)
  H <- hypothesis(brmsModel, "b_ClassMulti_emb - b_ClassID_emb > 0", class = NULL)
  H <- hypothesis(brmsModel, "b_ClassID_emb - b_ClassMulti_emb > 0", class = NULL)


  
  colMeans(H$samples > 0)
  
  featureSpaces <- c("Texture","Shape","pixelPCA","ShapeANDTexture","ShapeANDpixelPCA", 
                       "Triplet_emb","ClassID_emb","ClassMulti_emb","AE_emb","viAE_emb","VAE_emb","beta10VAE_emb", 
                       "ShapeANDAE_emb","ShapeANDviAE_emb","ShapeANDbeta1VAE_emb","ShapeANDClassMulti_embANDbeta1VAE",
                       "ShapeANDTextureANDAE_emb","ShapeANDTextureANDviAE_emb", 
                       "Texture_delta","Shape_delta","pixelPCA_delta","Triplet_delta","ClassID_delta",
                       "ClassMulti_delta","AE_delta","viAE_delta","beta1VAE_delta",
                       "ShapeVertex_delta","TexturePixel_delta", 
                       "Texture_deltalincomb","Shape_deltalincomb","pixelPCA_deltalincomb","Triplet_deltalincomb", 
                       "ClassID_deltalincomb","ClassMulti_deltalincomb","AE_deltalincomb", 
                       "viAE_deltalincomb","beta1VAE_deltalincomb", 
                       "ClassID_dn","ClassMulti_dn","VAE_ldn","VAE_nldn")
  
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
  writeMat(paste(rootDir,destinFolder,"hypotheses_forwardModelKTAll.mat",sep = ""),evidenceRatios=evidenceRatios,pp=pp)
  