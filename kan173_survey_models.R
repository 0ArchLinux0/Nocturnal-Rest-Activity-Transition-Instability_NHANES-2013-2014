#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("usage: kan173_survey_models.R <model-input.csv> <output.csv>")
}
if (!requireNamespace("survey", quietly = TRUE)) {
  stop("R package 'survey' is unavailable")
}
suppressPackageStartupMessages(library(survey))
options(survey.lonely.psu = "adjust")
input_path <- args[[1L]]
output_path <- args[[2L]]
data <- read.csv(input_path, check.names = FALSE, stringsAsFactors = FALSE)

fit_one <- function(model_name, formula, required) {
  subset <- data[data$model == model_name, , drop = FALSE]
  required <- unique(c(required, "WTMEC2YR", "SDMVSTRA", "SDMVPSU"))
  missing <- setdiff(required, names(subset))
  if (length(missing) > 0L) {
    stop(paste("survey input missing columns:", paste(missing, collapse = ", ")))
  }
  complete <- complete.cases(subset[, required, drop = FALSE])
  complete <- complete & is.finite(subset$WTMEC2YR) & subset$WTMEC2YR > 0
  complete <- complete & is.finite(subset$SDMVSTRA) & is.finite(subset$SDMVPSU)
  subset <- subset[complete, , drop = FALSE]
  if (nrow(subset) <= length(required)) {
    stop(paste("insufficient survey complete cases for", model_name))
  }
  subset$Sex <- factor(subset$Sex)
  design <- svydesign(
    ids = ~SDMVPSU,
    strata = ~SDMVSTRA,
    weights = ~WTMEC2YR,
    nest = TRUE,
    data = subset
  )
  fit <- svyglm(formula, design = design, family = gaussian())
  coefficients <- summary(fit)$coefficients
  estimate <- coefficients["P01_10m", "Estimate"]
  se <- coefficients["P01_10m", "Std. Error"]
  p_column <- grep("^Pr\\(", colnames(coefficients), value = TRUE)
  p_value <- coefficients["P01_10m", p_column[[1L]]]
  degrees <- degf(design)
  critical <- qt(0.975, df = degrees)
  data.frame(
    model = model_name,
    N = nrow(subset),
    beta_P01 = as.numeric(estimate),
    SE_P01 = as.numeric(se),
    CI95_low = as.numeric(estimate - critical * se),
    CI95_high = as.numeric(estimate + critical * se),
    p_P01 = as.numeric(p_value),
    design_df = as.numeric(degrees),
    weight = "WTMEC2YR",
    strata = "SDMVSTRA",
    PSU = "SDMVPSU",
    stringsAsFactors = FALSE
  )
}

model1 <- fit_one(
  "Model 1",
  PHQ9_Score ~ P01_10m + Age_z + factor(Sex) + BMI_z,
  c("PHQ9_Score", "P01_10m", "Age_z", "Sex", "BMI_z")
)
model2 <- fit_one(
  "Model 2",
  PHQ9_Score ~ P01_10m + Age_z + factor(Sex) + BMI_z + PIR_z +
    self_reported_weekday_sleep_hours_z + WASO_proxy_min_z,
  c(
    "PHQ9_Score", "P01_10m", "Age_z", "Sex", "BMI_z", "PIR_z",
    "self_reported_weekday_sleep_hours_z", "WASO_proxy_min_z"
  )
)
write.csv(rbind(model1, model2), output_path, row.names = FALSE, na = "")
cat("Survey-weighted svydesign/svyglm models written to", output_path, "\n")
