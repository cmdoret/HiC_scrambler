# simulates a hic matrix

import numpy as np

# to generate NB component of the cell value
.sim.NBvalue <- function(Bmean, r = 10) {
  value <- rnbinom(1, mu = Bmean, size = r)
  return(value)
}


def powerlaw(dist, C, alpha):
    if distance == 0:
        return C
    return C * dist**(-alpha)

def normal_bias(distance):
  return (1 + np.exp(-((distance - 20)^2)/(2 * 30))) * 4


# function to add bias to a matrix
def add_bias(M, slope, medianIF, powerlaw_alpha, biasFun=normal_bias
  col_num = 0
  for i in range(M.shape[0]):
    for j in range(col_num, M.shape[1]):
      distance = j - i
      Bmean = powerlaw(distance, medianIF, powerlaw_alpha)
      new_value = np.round(M[i, j] * biasFunc(distance))
      new_value = 0 if new_value < 0 else new_value
      mat[i, j] <- new_value
      mat[j, i] <- mat[i, j]
    }
    col_num <- col_num + 1
  }
  return(mat)
}

# simulate matrix function will create a two full contact maps.
# Matrices will have same signal component but different noise
# components
.sim.mat <- function(nrow = 100, ncol = 100, medianIF, sdIF, powerlaw.alpha,
                     sd.alpha, prop.zero.slope) {
  # check for invalid proportion of zeros slope
  if (.prop.zero.linear(prop.zero.slope, nrow - 1) > 1) {
    stop("prop.zero.slope is too large and will produce probabilities
         greater than 1 at the maximum distance in the matrix.")
  }
  cell1 <- matrix(nrow = nrow, ncol = ncol)
  cell2 <- matrix(nrow = nrow, ncol = ncol)
  col_num <- 1

  j <- unlist(Map(seq, 1:ncol(cell1), MoreArgs=list(ncol(cell1))))
  i <- rep(seq_len(nrow(cell1)), times=head(rev(seq_len(ncol(cell1))), nrow(cell1)))
  distance <- j - i + 1
  Bmean <- .powerlaw(distance, medianIF, powerlaw.alpha)
  noise.sd <- .powerlaw(distance, sdIF, sd.alpha)
  idx <- cbind(i, j)
  cell1[idx] <- round(Bmean) + round(rnorm(i, 0, noise.sd))
  cell2[idx] <- round(Bmean) + round(rnorm(i, 0, noise.sd))
  prob.zero <- .prop.zero.linear(distance, prop.zero.slope)
  u1 <- runif(length(prob.zero))
  u2 <- runif(length(prob.zero))
  zero_index1 <- ifelse(u1 < prob.zero, TRUE, FALSE)
  zero_index2 <- ifelse(u2 < prob.zero, TRUE, FALSE)
  cell1[idx[zero_index1,]] <- 0
  cell2[idx[zero_index2,]] <- 0
  idx2 <- idx[, c(2,1)]
  cell1[idx2] <- cell1[idx]
  cell2[idx2] <- cell2[idx]
  # check for negative values
  cell1[cell1 < 0] <- 0
  cell2[cell2 < 0] <- 0
  return(list(cell1, cell2))
}

# default bias functions


.no.bias <- function(distance) {
  1
}



# add true differences to the matrices
.sim.differences <- function(mat1, mat2, fold.change = 2,
                             i.range, j.range) {
  for (n in 1:length(i.range)) {
    i <- i.range[n]
    j <- j.range[n]
    # get which direction the difference is
    diff_direction <- sign(mat1[i, j] - mat2[i, j])
    newIF <- as.integer(round(mat1[i, j] * fold.change^diff_direction))
    mat1[i, j] <- ifelse(newIF < 1, 1, newIF)
    mat1[j, i] <- mat1[i, j]
  }
  return(mat1)
}


# add CNV
.add.CNV <- function(mat, CNV.location, CNV.proportion, CNV.multiplier) {
  loc <- CNV.location[1]:CNV.location[2]
  idx <- expand.grid(loc, loc)
  n.idx <- nrow(idx)
  new.idx <- sample(1:n.idx, size = round(CNV.proportion * n.idx), replace = FALSE)
  new.idx <- idx[new.idx,]
  new.idx <- as.matrix(new.idx)
  mat[new.idx] <- mat[new.idx] * CNV.multiplier
  # force symmetry
  mat[lower.tri(mat)] <- t(mat)[lower.tri(mat)]
  return(mat)
}

# wrapper function for simulation studies will generate the matrices
# and perform HiCcompare analysis on them

#' Simulate a Hi-C matrix and perform HiCcompare analysis on it
#'
#' @export
#' @param nrow Number of rows and columns of the full matrix
#' @param medianIF The starting value for a power law distribution
#'     for the interaction frequency of the matrix. Should use the median
#'     value of the IF at distance = 0. Typical values for 1MB data
#'     are around 50,000.
#'     For 500kb data typical values are 25,000. For 100kb data, 4,000.
#'     For 50kb data, 1,800.
#' @param sdIF The estimated starting value for a power law distriubtion
#'     for the standard deviaton of the IFs. Should use the SD of the IF at
#'     distance = 0. Typical value for 1MB data is 19,000.
#' @param powerlaw.alpha The exponential parameter for the power law
#'     distribution for the median IF. Typical values are 1.6 to 2.
#'     Defaults to 1.8.
#' @param sd.alpha The exponential parameter for the power law
#'     distribution for the SD of the IF. Typical values are 1.8 to 2.2.
#'     Defaults to 1.9.
#' @param prop.zero.slope The slope to be used for a linear function of
#'     the probability of zero in matrix = slope * distance
#' @param centromere.location The location for a centromere to be
#'     simulated. Should be entered as a vector of 2 numbers; the
#'     start column number and end column number. i.e. to put a centromere
#'     in a 100x100 matrix starting at column 47 and ending at column 50
#'     enter centromere.location = c(47, 50). Defaults NA indicating no
#'     simulated centromere will be added to the matrix.
#' @param CNV.location The location for a copy number variance (CNV).
#'     Should be entered as a vector of 2 numbers; the
#'     start column number and end column number. i.e. to put a CNV
#'     in a 100x100 matrix starting at column 1 and ending at column 50
#'     enter CNV.location = c(1, 50). Defaults NA indicating no
#'     simulated CNV will be added to the matrices. If a value is
#'     entered one of the matrices will have a CNV applied to it.
#' @param CNV.proportion The proportion of 0's to be applied to the
#'    CNV location specified. Defaults to 0.8.
#' @param CNV.multiplier A multiplyer to be applied as the CNV. To
#'     approximate deletion set to 0, to increase copy numbers set to
#'     a value > 1. Defaults to 0.
#' @param biasFunc A function used for adding bias to one of the simulated
#'     matrices. Should take an input of unit distance and generally have
#'     the form of 1 + Probability Density Function with unit distance as the
#'     random variable. Can also use a constant as a scaling factor
#'     to add a global offset to one of the matrices. The output of the bias
#'     function will be multiplied to the IFs of one matrix.
#'     Included are a normal kernel bias and a no bias function. If no function
#'     is entered, a normal kernel bias with an additional global scaling
#'     factor of 4 will be used. To use no bias set biasFunc = .no.bias, see
#'     examples section.
#' @param fold.change The fold change you want to introduce for true differences
#'     in the simulated matrices. Defaults to NA for no fold change added.
#' @param i.range The row numbers for the cells that you want to introduce true
#'     differences at. Must be same length as j.range.
#'     Defaults to NA for no changes added.
#' @param j.range The column numbers for the cells that you want to introduce
#'     true differences at.  Must be same length as
#'     Defaults to NA for no changes added.
#' @param Plot Logical, should the HiCdiff plots be output? Defaults to TRUE.
#' @param scale Logical, Should scaling be applied for the HiCdiff procedure?
#'     Defaults to TRUE.
#' @param alpha Type I error rate parameter. At what level should a significant
#'     difference be defined. Defaults to 0.05.
#' @param diff.thresh Parameter for hic_diff procedure. See ?hic_diff for more
#'     help. Defaults to NA.
#' @param include.zeros Should partial zero interactions be included? Defaults
#'     to FALSE.
#'
#' @return A list containing the true positive rate (TPR), the specificity (SPC),
#'     the p-values, the hic.table object, true differences - a data.table
#'     of the rows of the hic.table where a true difference was applied, the truth
#'     vector - a vector of 0's and 1's where 1 indicates a true
#'     difference was applied to that cell, sim.table - the hic.table object for
#'     the simulate matrices before hic_loess and hic_compare was run on it.
#'
#' @examples
#' # simulate two matrices with no fold changes introduced using default values
#' sim <- hic_simulate()
#'
#' # example of bias functions
#' ## the default function used
#' .normal.bias = function(distance) {
#'   (1 + exp(-((distance - 20)^2) / (2*30))) * 4
#' }
#'
#' ## an additional bias function
#' .no.bias = function(distance) {
#'   1
#' }
#'
#' # simulate matrices with 200 true differences using no bias
#' i.range = sample(1:100, replace=TRUE)
#' j.range = sample(1:100, replace=TRUE)
#' sim2 <- hic_simulate(nrow=100, biasFunc = .no.bias, fold.change = 5,
#'                      i.range = i.range, j.range = j.range)
#'
#'
hic_simulate <- function(nrow = 100, medianIF = 50000, sdIF = 14000,
                         powerlaw.alpha = 1.8,
                         sd.alpha = 1.9, prop.zero.slope = 0.001,
                         centromere.location = NA,
                         CNV.location = NA,
                         CNV.proportion = 0.8,
                         CNV.multiplier = 0,
                         biasFunc = .normal.bias, fold.change = NA,
                         i.range = NA, j.range = NA, Plot = TRUE,
                         scale = TRUE, alpha = 0.05,
                         diff.thresh = NA,
                         include.zeros = FALSE) {

  if (is.na(fold.change) & (is.na(i.range[1]) | is.na(j.range[1]))) {
    i.range <- 1
    j.range <- 1
  }
  if (!is.na(fold.change) & (is.na(i.range[1]) | is.na(j.range[1]))) {
    stop("Error: Please enter values for i.range and j.range if
         you wish to produce a fold change in the simulated matrix")
  }
  if (!is.na(centromere.location[1])) {
    if (length(centromere.location) != 2 | !is.numeric(centromere.location)) {
      stop('Centromere location should be a vector of 2 numbers')
    }
    if (centromere.location[1] < 0 | centromere.location[1] > nrow | centromere.location[2] > nrow) {
      stop('centromere.location is outside the bounds of the matrix')
    }
    if (sum(i.range %in% centromere.location[1]:centromere.location[2] +
        j.range %in% centromere.location[1]:centromere.location[2] == 2)) {
      stop('enter i.range and j.range that does not include changes within the centromere')
    }
  }
  if (!is.na(CNV.location[1])) {
    if (length(CNV.location) != 2 | !is.numeric(CNV.location)) {
      stop('CNV.location should be a vector of 2 numbers')
    }
    if (centromere.location[1] < 0 | centromere.location[1] > nrow | centromere.location[2] > nrow) {
      stop('CNV.location is outside the bounds of the matrix')
    }
  }
  ncol <- nrow
  # simulate matrices
  sims <- .sim.mat(nrow, ncol, medianIF, sdIF, powerlaw.alpha, sd.alpha,
                   prop.zero.slope)

  # if fold.change = NA no true differences will be added to the matrix
  if (!is.na(fold.change)) {
    # make sure no cells are duplicated
    temp.tab <- data.frame(i = i.range, j = j.range)
    temp.tab <- unique(temp.tab)
    i.range <- temp.tab$i
    j.range <- temp.tab$j
    diff <- .sim.differences(sims[[2]], sims[[1]], fold.change, i.range,
                             j.range)
    sims[[2]] <- diff
  }
  # add in sample specific bias to one matrix
  sims[[1]] <- .add.bias(sims[[1]], bias.slope, medianIF, powerlaw.alpha,
                         biasFunc = biasFunc)

  # add centromere to matrix
  if (!is.na(centromere.location[1])) {
    cent <- centromere.location[1]:centromere.location[2]
    sims[[1]][cent,] <- 0
    sims[[1]][, cent] <- 0
    sims[[2]][, cent] <- 0
    sims[[2]][cent,] <- 0
  }

  # add CNV to matrix
  if (!is.na(CNV.location[1])) {
    sims[[1]] <- .add.CNV(sims[[1]], CNV.location, CNV.proportion, CNV.multiplier)
    #sims[[2]] <- .add.CNV(sims[[2]], CNV.location, CNV.proportion, CNV.multiplier)
  }
  # perform HiCloess on simulated data convert matrix to sparse format
  colnames(sims[[1]]) <- 1:nrow
  colnames(sims[[2]]) <- 1:nrow
  sims[[1]] <- full2sparse(sims[[1]])
  sims[[2]] <- full2sparse(sims[[2]])
  backup.sim.table <- create.hic.table(sims[[1]], sims[[2]], chr = "ChrSim",
                                       scale = FALSE,
                                       include.zeros = include.zeros)
  sims <- create.hic.table(sims[[1]], sims[[2]], chr = "ChrSim",
                           scale = scale, include.zeros = include.zeros)
  #####
  # ORIGINAL CONFIGURATION WHEN USING HIC_DIFF()
  # normed <- hic_loess(sims, Plot = Plot, diff.thresh = diff.thresh,
  #                     check.differences = TRUE)
  # pvals <- normed$p.value
  ####
  ####
  # NEW CONFIGURATION WHEN USING HIC_COMPARE()
  normed <- hic_loess(sims, Plot = Plot)
  normed <- hic_compare(normed, Plot = Plot)
  pvals <- normed$p.adj
  ####

  # get the true differences
  true.diffs <- data.table(i = c(i.range, j.range), j = c(j.range, i.range))
  true.diffs <- left_join(true.diffs, normed, by = c(i = "start1", j = "start2"))
  # remove duplicated rows
  true.diffs <- true.diffs[!duplicated(true.diffs), ]
  true.diffs <- as.data.table(na.omit(true.diffs))

  # calculate sensitivty and specificity
  true.pos <- sum(true.diffs$p.value < alpha)
  false.pos <- sum(normed$p.value < alpha, na.rm = TRUE) - true.pos
  false.neg <- nrow(true.diffs) - true.pos
  true.neg <- nrow(normed) - true.pos - false.pos - false.neg
  TPR <- true.pos/(true.pos + false.neg)
  SPC <- true.neg/(true.neg + false.pos)

  # make vectors to feed into ROC packages vector of p-value decision

  temp.true.diffs <- true.diffs[, `:=`(truth, 1)]
  temp.true.diffs <- temp.true.diffs[, c("i", "j", "truth"), with = FALSE]
  truth <- left_join(normed, temp.true.diffs, by = c(start1 = "i", start2 = "j"))
  truth$truth[is.na(truth$truth)] <- 0
  truth <- truth$truth
  
  #### 
  # OLD VERSION FOR HIC_DIFF()
  # results <- list(TPR = TPR, SPC = SPC, pvals = normed$p.value, hic.table = normed,
                  # true.diff = true.diffs, truth = truth, sim.table = backup.sim.table)
  ####
  # NEW VERSION FOR HIC_COMPARE()
  results <- list(TPR = TPR, SPC = SPC, pvals = normed$p.adj, hic.table = normed,
                  true.diff = true.diffs, truth = truth, sim.table = backup.sim.table)
  ####
  
  if (!is.na(fold.change)) {
    message("True Positives: ", true.pos, " Total added differences: ",
                nrow(true.diffs), " True Negatives: ", true.neg, sep = "")
    message("TPR: ", TPR, sep = "")
    message("SPC: ", SPC, sep = "")
  }
  return(results)
}


# function to get TPR/SPC and generate truth matrix results of HiCcompare
# on non-hicloess normalized data enter sim.table as a hic.table for
# the pre-normalized simulated matrices. Normalization can be done
# using some other method than hic_loess

#' Compare other normalization methods on simulated data
#'
#' @export
#' @param sim.table the sim.table object output from hic_simulate
#' @param i.range The row numbers for the cells that you want to introduce
#'     true differences at. Must be same length as j.range.
#' @param j.range The column numbers for the cells that you want to introduce
#'     true differences at.  Must be same length as i.range.
#' @param Plot Logical, should the HiCdiff plots be output? Defaults to TRUE.
#' @param alpha Type I error rate parameter. At what level should a significant
#'     difference be defined. Defaults to 0.05.
#' @param diff.thresh Parameter for hic_diff procedure. see ?hic_diff for
#'     more details.
#'
#' @return A list containing the true positive rate (TPR), the specificity (SPC),
#'     the p-values, the hic.table object, true differences - a data.table
#'     of the rows of the hic.table where a true difference was applied, the truth
#'     vector - a vector of 0's and 1's where 1 indicates a true
#'     difference was applied to that cell.
#' @examples
#' i.range = sample(1:100, replace=TRUE)
#' j.range = sample(1:100, replace=TRUE)
#' sim <- hic_simulate(i.range = i.range, j.range = j.range, fold.change = 2)
#' mat1 <- sim$sim.table[, c('start1', 'start2', 'IF1'), with=FALSE]
#' mat2 <- sim$sim.table[, c('start1', 'start2', 'IF2'), with=FALSE]
#' mat1 <- sparse2full(mat1) %>% KRnorm
#' mat2 <- sparse2full(mat2) %>% KRnorm
#' colnames(mat1) <- 1:ncol(mat1)
#' colnames(mat2) <-1:ncol(mat2)
#' mat1 <- full2sparse(mat1)
#' mat2 <- full2sparse(mat2)
#' new.tab <- create.hic.table(mat1, mat2, chr= 'chrsim')
#' sim2 <- sim.other.methods(new.tab, i.range = i.range , j.range = j.range)
#'
sim.other.methods <- function(sim.table, i.range, j.range, Plot = TRUE,
                              alpha = 0.05, diff.thresh = NA) {
  if (ncol(sim.table) < 11) {
    sim.table <- sim.table[, `:=`(adj.IF1, IF1)]
    sim.table <- sim.table[, `:=`(adj.IF2, IF2)]
    sim.table <- sim.table[, `:=`(adj.M, M)]
  }
  ####
  # OLD VERSION FOR HIC_DIFF()
  # diffs <- hic_diff(sim.table, Plot = Plot, diff.thresh = diff.thresh)
  # pvals <- diffs$p.value
  ####
  # NEW VERSION FOR HIC_COMPARE()
  diffs <- hic_compare(sim.table, Plot = Plot)
  pvals <- diffs$p.adj
  ####

  true.diffs <- data.table(i = c(i.range, j.range), j = c(j.range, i.range))
  true.diffs <- left_join(true.diffs, diffs, by = c(i = "start1", j = "start2"))
  # remove duplicated rows
  true.diffs <- true.diffs[!duplicated(true.diffs), ]
  true.diffs <- as.data.table(na.omit(true.diffs))

  # calculate sensitivty and specificity
  ####
  # OLD VERSION FOR HIC_DIFF()
  # true.pos <- sum(true.diffs$p.value < alpha)
  # false.pos <- sum(diffs$p.value < alpha, na.rm = TRUE) - true.pos  #### changed to < 0.1
  ####
  # NEW VERSION FOR HIC_COMPARE()
  true.pos <- sum(true.diffs$p.adj < alpha)
  false.pos <- sum(diffs$p.adj < alpha, na.rm = TRUE) - true.pos  #### changed to < 0.1
  ####
  false.neg <- nrow(true.diffs) - true.pos
  true.neg <- nrow(diffs) - true.pos - false.pos - false.neg
  TPR <- true.pos/(true.pos + false.neg)
  SPC <- true.neg/(true.neg + false.pos)

  # make vectors to feed into ROC packages vector of p-value decision
  temp.true.diffs <- true.diffs[, `:=`(truth, 1)]
  temp.true.diffs <- temp.true.diffs[, c("i", "j", "truth"), with = FALSE]
  truth <- left_join(diffs, temp.true.diffs, by = c(start1 = "i", start2 = "j"))
  truth$truth[is.na(truth$truth)] <- 0
  truth <- truth$truth

  ####
  # OLD VERSION FOR HIC_DIFF()
  # results <- list(TPR = TPR, SPC = SPC, pvals = diffs$p.value, hic.table = diffs,
                  # true.diff = true.diffs, truth = truth)
  ####
  # NEW VERSION FOR HIC_COMPARE()
  results <- list(TPR = TPR, SPC = SPC, pvals = diffs$p.adj, hic.table = diffs,
                  true.diff = true.diffs, truth = truth)
  ####
  message("True Positives: ", true.pos, " Total added differences: ",
              nrow(true.diffs), " True Negatives: ", true.neg, sep = "")
  message("TPR: ", TPR, sep = "")
  message("SPC: ", SPC, sep = "")
  return(results)
}
