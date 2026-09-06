##############################################@

#CONVOLUTIONAL NEURAL NETWORK (CNN) SCRIPT FOR HEDERA LEAF MASKS (v. APRIL14)----

##############################################@

#0.INFO----

#Project: TFM_Hedera
#Collaborators (co-directors): Virginia Valcárcel, Anaïs Gibert
#Goal: to test the ability of image classification in identification of Hedera species
#Objectives
# 
# To test whether different species of Hedera can be identified using image based classifciation methods of leaves.
# We aim to use different approach from supervised and unsupervised methods, with different level of data extraction required.
# We plan to focus mainly on leaves shape, and maybe on some specific traits, but not color.
#In this script, we aim to understand how different values in the data augmentation layers affect the model accuracy
#Script is run by executing commands in point 6. The previous lineas are for establishing the variations on the base script needed to run the different models.
#1. INSTALLING THE PACKAGES ----

install.packages(c("imager", "magrittr", "dplyr", "stringr", "tibble", "fda",
                   "MFPCA", "FactoMineR", "mvtnorm", "Momocs", "tidyverse", "tibble",
                   "dplyr","stringr","fs","tools" , "tensorflow", "patchwork", "keras3",
                   "magick"))

install.packages("reticulate")
reticulate::install_python("3.11")
keras3::install_keras()

#2. LOADING THE LIBRARIES----

library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)  # to combine graphs

library(tibble)
library(dplyr)
library(stringr)
library(tools)

library(keras3)
library(tensorflow)
library(fs)
library(reticulate)

library(magick)


#3. CNN MODEL WITH FLIPPING----

#Everything here is overwritten with the variations in 6.
CNN_model_flipping <- function(model_name = "El_model",
                          src_dir = "data/leaf_masks_nuevas", #source directory with leaf masks
                          out_dir = paste0("data/", model_name, "/") , # output file with train/valid/test
                          out_dir_figure = paste0("data/", model_name, "/figures/") ,
                          p_train  = 0.8, 
                          p_valid  = 0.1,            # remaining = test
                          copy_files = TRUE,             # TRUE = copy, FALSE = move
                          target = "sp",              # "sp" ou "leaftype" ou "sp_leaftype"
                          batch_size = 32,
                          img_size = c(224, 224),
                          erasing = 0.1, 
                          zoom = 0.1,
                          rotation = 0.1,
                          translation = 0.1
                            ){
  
##3.1. Set up model parameters

set.seed(30)

#Building the out_dir (if necessary)
# dir_delete(out_dir)
dir_create(out_dir)
# dir_delete(out_dir_figure)
dir_create(out_dir_figure)

##3.2. Build metadata (FORMAT: AZO_01SRC25(1)_1_RP.jpg)

#List all images in src_dir
files_all <- dir_ls(src_dir, type = "file", glob = "*.jpg")

#meta: data frame containing the images' paths + name decomposition (used in image sorting)
meta <- tibble(path = files_all) %>%
  mutate(
    file = file_path_sans_ext(path_file(path)),
    ext  = file_ext(path),
    m = str_match(file, "^([A-Z]+)_([^()]+)\\((\\d+)\\)_(\\d+)_([A-Z]+)$")
  ) %>%
  mutate(
    sp        = m[,2],
    pop       = m[,3],
    ind       = as.integer(m[,4]),
    numbranch = as.integer(m[,5]),
    leaftype  = m[,6]
  ) %>%
  dplyr::select(-m) %>%
  filter(!is.na(sp), !is.na(leaftype))

#Creating the "class_dir" variable inside the "meta" data frame, used in the next step
meta <- meta %>%
  mutate(
    class = if (target == "sp") {
      sp
    } else if (target == "leaftype") {
      leaftype
    } else if (target == "sp_leaftype") {
      paste(sp, leaftype, sep = "__")
    } else {
      sp
    }
  )

meta <- meta %>%
  mutate(
    class_dir = class %>%
      as.character() %>%
      stringr::str_to_lower() %>%
      stringr::str_replace_all("[^a-z0-9]+", "_") %>%
      stringr::str_replace_all("^_|_$", ""),
    gid = paste(sp, pop, ind, sep = "__")   
  )

#3.3. Split images as training, test, validation

#Here stratified (80/10/10 by class) with attributing all picture of one individual to the same dataset
set.seed(30)

#Table of individuals and their correspondent species:
group_df <- meta %>%
  dplyr::distinct(gid, class_dir)

#split_groups_one_class: each individual (and, in consequence, their leaves) is assigned one dataset partition 
split_groups_one_class <- function(df, p_train, p_valid) {
  n <- nrow(df)
  idx <- sample.int(n)
  
  n_train <- floor(p_train * n)
  n_valid <- floor(p_valid * n)
  
  df %>%
    mutate(.split = dplyr::case_when(
      row_number() %in% idx[1:n_train] ~ "train",
      row_number() %in% idx[(n_train + 1):(n_train + n_valid)] ~ "valid",
      TRUE ~ "test"
    ))
}

group_split <- group_df %>%
  dplyr::group_by(class_dir) %>%
  dplyr::group_modify(~ split_groups_one_class(.x, p_train, p_valid)) %>%
  dplyr::ungroup()

#Attaching the result of split_groups_one_class to the data frame "meta"
meta_split <- meta %>%
  dplyr::left_join(group_split, by = c("gid", "class_dir"))

cat("=== Dataset spliting ===\n")
print(meta_split %>% count(.split, class_dir))

#Assigning the images its correspondent dataset partition according to the .split column in meta_split (created with the split_groups_one_class function)
splits <- c("train", "valid", "test")
classes <- sort(unique(meta_split$class_dir))

for (sp in splits) {
  for (cl in classes) {
    dir_create(path(out_dir, sp, cl), recurse = TRUE)
  }
}

#move_or_copy: function in charge of moving the images
move_or_copy <- function(src, dest, do_copy = TRUE) {
  if (!file_exists(src)) return(FALSE)
  
  # avoid name collision
  dest_final <- dest
  if (file_exists(dest_final)) {
    ext <- path_ext(dest_final)
    base <- path_ext_remove(path_file(dest_final))
    parent <- path_dir(dest_final)
    i <- 1
    repeat {
      candidate <- path(parent, paste0(base, "_", i, ifelse(ext == "", "", paste0(".", ext))))
      if (!file_exists(candidate)) { dest_final <- candidate; break }
      i <- i + 1
    }
  }
  
  if (do_copy) file_copy(src, dest_final, overwrite = FALSE) else file_move(src, dest_final)
  TRUE
}

results <- meta_split %>%
  mutate(
    filename = path_file(path),
    dest = path(out_dir, .split, class_dir, filename),
    ok = mapply(move_or_copy, path, dest, MoreArgs = list(do_copy = copy_files))
  )

#Checking all images are in their correspondent folders inside out_dir
cat("Source files missing:", sum(!results$ok), "\n")
print(results %>% count(.split, class_dir))

#.csv file for meta_split (name of the leaves, their dataset partition and the check we run last step)
write.csv(results, paste0(out_dir, model_name, "_dataset_CNN.csv"))

#3.4. Keras datasets

seed <- 123

train_ds <- image_dataset_from_directory(
  path(out_dir, "train"),
  image_size = img_size,
  batch_size = batch_size,
  seed = seed
)

valid_ds <- image_dataset_from_directory(
  path(out_dir, "valid"),
  image_size = img_size,
  batch_size = batch_size,
  seed = seed
)

#Defining the name and quantity of our classes (used for class weights)
class_names <- train_ds$class_names
num_classes <- length(class_names)
cat("=== Classes analyzed ===\n")
print(class_names)

#3.5. Shufflining and prefetching
#Image shuffling allows more randomness when training the model.
#Prefetching speeds up the process of training the model by preloading n number of images from which the model will use whatever number is batch_size in the next step of the training
#We are going to let TensorFlow decide how many images to preload by using the AUTOTUNE feature

AUTOTUNE <- tf$data$AUTOTUNE

train_ds <- train_ds %>%
  tf$data$Dataset$shuffle(buffer_size = as.integer(1000L)) %>%
  tf$data$Dataset$prefetch(buffer_size = AUTOTUNE)

valid_ds <- valid_ds %>%
  tf$data$Dataset$prefetch(buffer_size = AUTOTUNE)

#3.6. Class weights (way to deal with image quantity/species desequilibrium)

train_counts <- results %>%
  filter(.split == "train") %>%
  count(class_dir)

#Inverse weight: more rare  => bigger weights
w_by_classdir <- with(train_counts, setNames(max(n) / n, class_dir))

#Mapper on the order of  class_names (order of the files)
w_vec <- as.numeric(w_by_classdir[class_names])
class_weight <- as.list(w_vec)

names(class_weight) <- as.character(0:(num_classes - 1))

cat("=== Weighted classification selected by the model (for unbalanced data) ===\n")
print(class_weight)

#3.1. Data augmentation (FLIP)  
#The first layers of the model will focus on data augmentation: flipping, rotating and zooming on the images to include even more randomness into the learning process
cat("Step of data augmentation\n")
data_augmentation <- keras_model_sequential() %>%
  layer_random_erasing(erasing) %>%
  layer_random_zoom(zoom) %>%
  layer_random_rotation(rotation) %>%
  layer_random_translation(translation, translation) %>%
  layer_random_flip("horizontal")

#3.8. Running the model
#(MobileNetV2 + augmentation of the data) + FIT with class_weight
cat("Model run  == MobileNetV2\n")
base_model <- application_mobilenet_v2(
  input_shape = c(img_size, 3),
  include_top = FALSE,
  weights = "imagenet"
)
base_model$trainable <- FALSE

inputs <- layer_input(shape = c(img_size, 3))

#Adding a pixel color value reescaling and a global_average_pooling_2d layer: transforming our images into two-dimensional data structures (numerical values between 0 and 1, organized in rows and columns)
#The "layer_dropout" layer helps with overfitting (adjusting our model too much to the "train" dataset, resulting in bad scores with the "validation" dataset) by dropping 20% of our training results.
x <- inputs %>%
  data_augmentation() %>%
  layer_rescaling(1/255) %>%
  base_model() %>%
  layer_global_average_pooling_2d() %>%
  layer_dropout(0.2)

#Finally, we add a layer in charge of adjusting our results into a probability distribution
outputs <- x %>% layer_dense(num_classes, activation = "softmax")
model <- keras_model(inputs, outputs)

model %>% compile(
  optimizer = optimizer_adam(learning_rate = 1e-3),
  loss = "sparse_categorical_crossentropy",
  metrics = c("accuracy")
)

#The "callbacks" object is used to stop the training once the learning_rate drops below a certain value. In this model, it also reduces the learning rate to a 20% if the learning rate stays the same for 2 epochs.
callbacks <- list(
  callback_early_stopping(patience = 5, restore_best_weights = TRUE),
  callback_reduce_lr_on_plateau(patience = 2, factor = 0.2)
)

history <- model %>% fit(
  train_ds,
  validation_data = valid_ds,
  epochs = 30,
  callbacks = callbacks,
  class_weight = class_weight
)

#3.9. Fine-tuning
cat("=== Start fine-tuning ===\n")

#Unlock backbone
base_model$trainable <- TRUE

#Recommended option: unlock only for the last layers
for (layer in base_model$layers[1:(length(base_model$layers) - 30)]) {
  layer$trainable <- FALSE
}

#Recompiled with a smaller learning rate 
model %>% compile(
  optimizer = optimizer_adam(learning_rate = 1e-5),
  loss = "sparse_categorical_crossentropy",
  metrics = "accuracy"
)

#Resume learning 
history_ft <- model %>% fit(
  train_ds,
  validation_data = valid_ds,
  epochs = 30,
  class_weight = class_weight,
  callbacks = callbacks
)

#3.10. Prediction for "test" dataset 

test_path <- fs::path(out_dir, "test")
stopifnot(fs::dir_exists(test_path))

#Predictions
test_ds_raw <- image_dataset_from_directory(
  test_path,
  image_size = img_size,
  batch_size = batch_size,
  shuffle = FALSE
)

#Names of classes
class_names <- test_ds_raw$class_names
k <- length(class_names)

# Predictions -> classes (1..k)
pred_probs <- model %>% predict(test_ds_raw)
y_pred <- max.col(pred_probs)  # 1..k

# Get True labels (y_true) from the dataset
y_true <- integer(0)
it <- test_ds_raw$as_numpy_iterator()
repeat {
  batch <- tryCatch(reticulate::iter_next(it), error = function(e) NULL)
  if (is.null(batch)) break
  y_true <- c(y_true, as.integer(batch[[2]]))
}

#Harmonize 0-based vs 1-based
# y_pred is 1..k ; y_true is 0..k-1
if (min(y_true) == 0) y_true <- y_true + 1

## To convert in label text
ref <- factor(class_names[y_true], levels = class_names)  # True
dat <- factor(class_names[y_pred], levels = class_names)  # Pred

###3.10.1 Confusion matrix and statistics

t_conf <- caret::confusionMatrix(dat, ref)
cat("=== Confusion Matrix and statitstics ===")
print(t_conf)

#Confusion matrix + accuracy
cm <- t_conf$table

write.csv(cm, paste0(out_dir, model_name, "_confusion.matrix.csv"))

###3.10.2 Metrics for the model (precision, recall, accuracy, F1 score)
  
  tp <- diag(cm)
  fp <- rowSums(cm) - tp
  fn <- colSums(cm) - tp
  
  precision <- tp / (tp + fp)
  recall <- tp / (tp + fn)
  
  f1 <- 2 * precision * recall / (precision + recall)
  
  macro_F1 <- mean(f1, na.rm = TRUE)
  macro_precision <- mean(precision, na.rm = TRUE)
  macro_recall <- mean(recall, na.rm = TRUE)
  macro_accuracy <- round((t_conf$overall[1] * 100), 2)
  
  
  # table by class: weighted F1
  f1_table <- data.frame(
    class = colnames(cm),
    precision = round(precision, 3),
    recall = round(recall, 3),
    F1 = round(f1, 3)
  )
  
  
  dataset_model <- data_frame(Model = model_name,
                              Accuracy = round((t_conf$overall[1] * 100), 2),
                              Accuracy_lower = round((t_conf$overall[3] * 100), 2),
                              Accuracy_Upper = round((t_conf$overall[4] * 100), 2),
                              Accuracy_null_model = round((t_conf$overall[5] * 100), 2),
                              Accuracy_pvalue = t_conf$overall[6],
                              Sensitivity = round((t_conf$byClass[1] * 100), 2),
                              Specificity = round((t_conf$byClass[2] * 100), 2),  # True Negative Rate
                              Precision = round((t_conf$byClass[3] * 100), 2),
                              Macro_F1=  round(mean(f1, na.rm = TRUE) * 100, 2),
                              Macro_recall = round(mean(recall, na.rm = TRUE)*100, 2))  

results <- list(metrics= dataset_model, F1byclass= f1_table, model_trained = model)


#3.11. Figures 

###3.11.1. Plotting accuracy/loss evolution through epochs
plot_history <- function(hist, title_prefix = "Training") {
  
  # df containing epochs, accuracy and loss from both first draft and fine-tuning 
  df <- data.frame(
    epoch = seq_along(hist$metrics$loss),
    train_acc = hist$metrics$accuracy,
    val_acc = hist$metrics$val_accuracy,
    train_loss = hist$metrics$loss,
    val_loss = hist$metrics$val_loss
  )
  
  # --- Accuracy ---
  acc_long <- df %>%
    dplyr::select(epoch, train_acc, val_acc) %>%
    pivot_longer(-epoch, names_to = "type", values_to = "value")
  
  p_acc <- ggplot(acc_long, aes(epoch, value, color = type)) +
    geom_line(linewidth = 1) +
    theme_minimal() +
    labs(
      title = paste(title_prefix, "Accuracy"),
      x = "Epoch",
      y = "Accuracy",
      color = ""
    )
  
  # --- Loss ---
  loss_long <- df %>%
    dplyr::select(epoch, train_loss, val_loss) %>%
    pivot_longer(-epoch, names_to = "type", values_to = "value")
  
  p_loss <- ggplot(loss_long, aes(epoch, value, color = type)) +
    geom_line(linewidth = 1) +
    theme_minimal() +
    labs(
      title = paste(title_prefix, "Loss"),
      x = "Epoch",
      y = "Loss",
      color = ""
    )
  
  #Combine both graphs:
  combined_plot <- p_acc / p_loss
  
  return(combined_plot)
}

p1 <- plot_history(history, "Initial Training")
p1

p2 <- plot_history(history_ft, "Fine-Tuning")
p2

#Saving the graphs for first draft (initial) and fine-tuning:
ggsave(paste0(out_dir_figure,"/Fig_training_curves_initial2.pdf"), p1, width = 8, height = 10)
ggsave(paste0(out_dir_figure,"/Fig_training_curves_finetune2.pdf"), p2, width = 8, height = 10)

###3.11.2. Heatmap of confusion matrix
cm_prop <- prop.table(cm, margin = 2)  # normalization by column ("True" class) 
cm_prop_df <- as.data.frame(cm_prop)

p1 <- ggplot(cm_prop_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%.2f", Freq)), size = 3) +
  theme_minimal() +
  labs(
    title = "Confusion matrix (normalized by true class)",
    x = "True label",
    y = "Predicted label"
  )

p1

ggsave(paste0(out_dir_figure, model_name,
  "_Fig_confusion_matrix.pdf"),
  plot = p1,
  width = 8,
  height = 6
)

###3.11.3. Unclassified images

#get_ordered_files: renaming all images as their address inside the "test" directory
get_ordered_files <- function(test_path, class_names) {
  files <- c()
  for (cl in class_names) {
    cl_dir <- fs::path(test_path, cl)
    # important: tri (pour coller à l'ordre du dataset)
    cl_files <- sort(fs::dir_ls(cl_dir, type = "file", glob = "*.jpg"))
    files <- c(files, cl_files)
  }
  files
}

test_files <- get_ordered_files(test_path, class_names)

#Checking all files have been labeled succesfully by get_ordered_files
cat("nb files:", length(test_files), "\n")
cat("nb labels  :", length(y_true), "\n")

#to_name: changes the numerical tags classes had in y_pred and y_true (as seen in point 5.1.) for their correspondent names from class_names
k <- length(class_names)
to_name <- function(y) {
  if (min(y) == 0) class_names[y + 1] else class_names[y]
}

true_names <- to_name(y_true)
pred_names <- to_name(y_pred)

#Search for misidentified leaves by checking where the y_pred and y_true tags don't match:
wrong_idx <- which(true_names != pred_names)
cat("Nb error:", length(wrong_idx), "\n")

#Search which leaves have been identified correctly
right_idx <- which(true_names == pred_names)


#show_misclassified: returns n misclassified leaves from the ones recoverd by wrong_idx, resizes them, and tags them with both the predicted and true class
show_misclassified <- function(n = wrong_idx, seed = 1, resize = "256x256") {
  set.seed(seed)
  idx <- sample(wrong_idx, min(n, length(wrong_idx)))
  
  imgs <- mapply(function(p, t, pr) {
    img <- image_read(p) |> image_resize(resize)
    image_annotate(
      img,
      text = paste0("True: ", t, "\nPred: ", pr),
      location = "+10+20",
      size = 20
    )
  }, test_files[idx], true_names[idx], pred_names[idx], SIMPLIFY = FALSE)
  
  image_montage(image_join(imgs), tile = paste0(ceiling(sqrt(length(imgs))), "x"))
}

#Show the misclassified images in a grid
img_grid <- show_misclassified(n = wrong_idx, seed = 42)

image_write(
  img_grid,
  path = paste0(out_dir_figure, model_name,
  "_Fig_misclassified_examples.pdf"),
  format = "pdf"
)

image_write(
  img_grid,
  path = paste0(out_dir_figure, model_name,
                "_Fig_misclassified_examples.png"),
  format = "png",
  density = 300
)

#show_confusion_pair: similar to show_misclassified, but showing all images missidentified the same way (they share both the true and the (wrongly) predicted class tag)
show_confusion_pair <- function(true_class, pred_class, n = wrong_idx) {
  idx <- which(true_names == true_class & pred_names == pred_class)
  if (length(idx) == 0) { cat("Aucune image pour ce couple.\n"); return(invisible(NULL)) }
  idx <- idx[1:min(n, length(idx))]
  
  imgs <- lapply(idx, function(i) {
    img <- image_read(test_files[i]) |> image_resize("256x256")
    image_annotate(img, text = fs::path_file(test_files[i]), location = "+10+20", size = 16)
  })
  
  image_montage(image_join(imgs), tile = paste0(ceiling(sqrt(length(imgs))), "x"))
}

p3 <- show_confusion_pair("hel", "hib", n = 3)

image_write(
  p3,
  path = paste0(out_dir_figure, model_name,
                "_fig_confusion_hel_to_hib.pdf"),
  format = "pdf"
)

p3

p4 <- show_confusion_pair("hib", "hel", n = 3)

image_write(
  p4,
  path = paste0(out_dir_figure, model_name,
                "_fig_confusion_hib_to_hel.pdf"),
  format = "pdf"
)
p4 

return(results)

###3.11.4. Lists of classified/misclassified images
test_files_nm <- str_remove(test_files, paste0(out_dir, "/test/([:alpha:]+)/"))

list_right <- c()
for (i in (right_idx)){
  list_right <- c(list_right, (test_files_nm[[i]]))
}

list_right <- as.data.frame.character(list_right)
write.csv(list_right, paste0(out_dir, "correct_list"))

list_wrong <- c()
for (i in (wrong_idx)){
  list_wrong <- c(list_wrong, (test_files_nm[[i]]))
}

list_wrong <- as.data.frame.character(list_wrong)
write.csv(list_wrong, paste0(out_dir, "wrong_list"))

###3.11.5. Prediction matrix
test_files_nm_df <- as.data.frame(test_files_nm)
pred_probs <- cbind(test_files_nm_df, pred_probs, row.names = NULL)
colnames(pred_probs) <- c("leaf", class_names)
write.csv(pred_probs, paste0(out_dir, "pred_probs"))
}

#4. CNN MODEL WITH NO FLIPPING----

#Everything here is overwritten with the variations in 6.
CNN_model_no_flipping <- function(model_name = "El_model",
                               src_dir = "data/leaf_masks_nuevas", #source directory with leaf masks
                               out_dir = paste0("data/", model_name, "/") , # output file with train/valid/test
                               out_dir_figure = paste0("data/", model_name, "/figures/") ,
                               p_train  = 0.8, 
                               p_valid  = 0.1,            # remaining = test
                               copy_files = TRUE,             # TRUE = copy, FALSE = move
                               target = "sp",              # "sp" ou "leaftype" ou "sp_leaftype"
                               batch_size = 32,
                               img_size = c(224, 224),
                               erasing = 0.1, 
                               zoom = 0.1,
                               rotation = 0.1,
                               translation = 0.1
){
  
  ##3.1. Set up model parameters
  
  set.seed(30)
  
  #Building the out_dir (if necessary)
  # dir_delete(out_dir)
  dir_create(out_dir)
  # dir_delete(out_dir_figure)
  dir_create(out_dir_figure)
  
  ##3.2. Build metadata (FORMAT: AZO_01SRC25(1)_1_RP.jpg)
  
  #List all images in src_dir
  files_all <- dir_ls(src_dir, type = "file", glob = "*.jpg")
  
  #meta: data frame containing the images' paths + name decomposition (used in image sorting)
  meta <- tibble(path = files_all) %>%
    mutate(
      file = file_path_sans_ext(path_file(path)),
      ext  = file_ext(path),
      m = str_match(file, "^([A-Z]+)_([^()]+)\\((\\d+)\\)_(\\d+)_([A-Z]+)$")
    ) %>%
    mutate(
      sp        = m[,2],
      pop       = m[,3],
      ind       = as.integer(m[,4]),
      numbranch = as.integer(m[,5]),
      leaftype  = m[,6]
    ) %>%
    dplyr::select(-m) %>%
    filter(!is.na(sp), !is.na(leaftype))
  
  #Creating the "class_dir" variable inside the "meta" data frame, used in the next step
  meta <- meta %>%
    mutate(
      class = if (target == "sp") {
        sp
      } else if (target == "leaftype") {
        leaftype
      } else if (target == "sp_leaftype") {
        paste(sp, leaftype, sep = "__")
      } else {
        sp
      }
    )
  
  meta <- meta %>%
    mutate(
      class_dir = class %>%
        as.character() %>%
        stringr::str_to_lower() %>%
        stringr::str_replace_all("[^a-z0-9]+", "_") %>%
        stringr::str_replace_all("^_|_$", ""),
      gid = paste(sp, pop, ind, sep = "__")   
    )
  
  #3.3. Split images as training, test, validation
  
  #Here stratified (80/10/10 by class) with attributing all picture of one individual to the same dataset
  set.seed(30)
  
  #Table of individuals and their correspondent species:
  group_df <- meta %>%
    dplyr::distinct(gid, class_dir)
  
  #split_groups_one_class: each individual (and, in consequence, their leaves) is assigned one dataset partition 
  split_groups_one_class <- function(df, p_train, p_valid) {
    n <- nrow(df)
    idx <- sample.int(n)
    
    n_train <- floor(p_train * n)
    n_valid <- floor(p_valid * n)
    
    df %>%
      mutate(.split = dplyr::case_when(
        row_number() %in% idx[1:n_train] ~ "train",
        row_number() %in% idx[(n_train + 1):(n_train + n_valid)] ~ "valid",
        TRUE ~ "test"
      ))
  }
  
  group_split <- group_df %>%
    dplyr::group_by(class_dir) %>%
    dplyr::group_modify(~ split_groups_one_class(.x, p_train, p_valid)) %>%
    dplyr::ungroup()
  
  #Attaching the result of split_groups_one_class to the data frame "meta"
  meta_split <- meta %>%
    dplyr::left_join(group_split, by = c("gid", "class_dir"))
  
  cat("=== Dataset spliting ===\n")
  print(meta_split %>% count(.split, class_dir))
  
  #Assigning the images its correspondent dataset partition according to the .split column in meta_split (created with the split_groups_one_class function)
  splits <- c("train", "valid", "test")
  classes <- sort(unique(meta_split$class_dir))
  
  for (sp in splits) {
    for (cl in classes) {
      dir_create(path(out_dir, sp, cl), recurse = TRUE)
    }
  }
  
  #move_or_copy: function in charge of moving the images
  move_or_copy <- function(src, dest, do_copy = TRUE) {
    if (!file_exists(src)) return(FALSE)
    
    # avoid name collision
    dest_final <- dest
    if (file_exists(dest_final)) {
      ext <- path_ext(dest_final)
      base <- path_ext_remove(path_file(dest_final))
      parent <- path_dir(dest_final)
      i <- 1
      repeat {
        candidate <- path(parent, paste0(base, "_", i, ifelse(ext == "", "", paste0(".", ext))))
        if (!file_exists(candidate)) { dest_final <- candidate; break }
        i <- i + 1
      }
    }
    
    if (do_copy) file_copy(src, dest_final, overwrite = FALSE) else file_move(src, dest_final)
    TRUE
  }
  
  results <- meta_split %>%
    mutate(
      filename = path_file(path),
      dest = path(out_dir, .split, class_dir, filename),
      ok = mapply(move_or_copy, path, dest, MoreArgs = list(do_copy = copy_files))
    )
  
  #Checking all images are in their correspondent folders inside out_dir
  cat("Source files missing:", sum(!results$ok), "\n")
  print(results %>% count(.split, class_dir))
  
  #.csv file for meta_split (name of the leaves, their dataset partition and the check we run last step)
  write.csv(results, paste0(out_dir, model_name, "_dataset_CNN.csv"))
  
  #3.4. Keras datasets
  
  seed <- 123
  
  train_ds <- image_dataset_from_directory(
    path(out_dir, "train"),
    image_size = img_size,
    batch_size = batch_size,
    seed = seed
  )
  
  valid_ds <- image_dataset_from_directory(
    path(out_dir, "valid"),
    image_size = img_size,
    batch_size = batch_size,
    seed = seed
  )
  
  #Defining the name and quantity of our classes (used for class weights)
  class_names <- train_ds$class_names
  num_classes <- length(class_names)
  cat("=== Classes analyzed ===\n")
  print(class_names)
  
  #3.5. Shufflining and prefetching
  #Image shuffling allows more randomness when training the model.
  #Prefetching speeds up the process of training the model by preloading n number of images from which the model will use whatever number is batch_size in the next step of the training
  #We are going to let TensorFlow decide how many images to preload by using the AUTOTUNE feature
  
  AUTOTUNE <- tf$data$AUTOTUNE
  
  train_ds <- train_ds %>%
    tf$data$Dataset$shuffle(buffer_size = as.integer(1000L)) %>%
    tf$data$Dataset$prefetch(buffer_size = AUTOTUNE)
  
  valid_ds <- valid_ds %>%
    tf$data$Dataset$prefetch(buffer_size = AUTOTUNE)
  
  #3.6. Class weights (way to deal with image quantity/species desequilibrium)
  
  train_counts <- results %>%
    filter(.split == "train") %>%
    count(class_dir)
  
  #Inverse weight: more rare  => bigger weights
  w_by_classdir <- with(train_counts, setNames(max(n) / n, class_dir))
  
  #Mapper on the order of  class_names (order of the files)
  w_vec <- as.numeric(w_by_classdir[class_names])
  class_weight <- as.list(w_vec)
  
  names(class_weight) <- as.character(0:(num_classes - 1))
  
  cat("=== Weighted classification selected by the model (for unbalanced data) ===\n")
  print(class_weight)
  
  #4.1. Data augmentation ( NO FLIP)  
  #The first layers of the model will focus on data augmentation: flipping, rotating and zooming on the images to include even more randomness into the learning process
  cat("Step of data augmentation\n")
  data_augmentation <- keras_model_sequential() %>%
    layer_random_erasing(erasing) %>%
    layer_random_zoom(zoom) %>%
    layer_random_rotation(rotation) %>%
    layer_random_translation(translation, translation)
    
  #3.8. Running the model
  #(MobileNetV2 + augmentation of the data) + FIT with class_weight
  cat("Model run  == MobileNetV2\n")
  base_model <- application_mobilenet_v2(
    input_shape = c(img_size, 3),
    include_top = FALSE,
    weights = "imagenet"
  )
  base_model$trainable <- FALSE
  
  inputs <- layer_input(shape = c(img_size, 3))
  
  #Adding a pixel color value reescaling and a global_average_pooling_2d layer: transforming our images into two-dimensional data structures (numerical values between 0 and 1, organized in rows and columns)
  #The "layer_dropout" layer helps with overfitting (adjusting our model too much to the "train" dataset, resulting in bad scores with the "validation" dataset) by dropping 20% of our training results.
  x <- inputs %>%
    data_augmentation() %>%
    layer_rescaling(1/255) %>%
    base_model() %>%
    layer_global_average_pooling_2d() %>%
    layer_dropout(0.2)
  
  #Finally, we add a layer in charge of adjusting our results into a probability distribution
  outputs <- x %>% layer_dense(num_classes, activation = "softmax")
  model <- keras_model(inputs, outputs)
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = 1e-3),
    loss = "sparse_categorical_crossentropy",
    metrics = c("accuracy")
  )
  
  #The "callbacks" object is used to stop the training once the learning_rate drops below a certain value. In this model, it also reduces the learning rate to a 20% if the learning rate stays the same for 2 epochs.
  callbacks <- list(
    callback_early_stopping(patience = 5, restore_best_weights = TRUE),
    callback_reduce_lr_on_plateau(patience = 2, factor = 0.2)
  )
  
  history <- model %>% fit(
    train_ds,
    validation_data = valid_ds,
    epochs = 30,
    callbacks = callbacks,
    class_weight = class_weight
  )
  
  #3.9. Fine-tuning
  cat("=== Start fine-tuning ===\n")
  
  #Unlock backbone
  base_model$trainable <- TRUE
  
  #Recommended option: unlock only for the last layers
  for (layer in base_model$layers[1:(length(base_model$layers) - 30)]) {
    layer$trainable <- FALSE
  }
  
  #Recompiled with a smaller learning rate 
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = 1e-5),
    loss = "sparse_categorical_crossentropy",
    metrics = "accuracy"
  )
  
  #Resume learning 
  history_ft <- model %>% fit(
    train_ds,
    validation_data = valid_ds,
    epochs = 30,
    class_weight = class_weight,
    callbacks = callbacks
  )
  
  #3.10. Prediction for "test" dataset 
  
  test_path <- fs::path(out_dir, "test")
  stopifnot(fs::dir_exists(test_path))
  
  #Predictions
  test_ds_raw <- image_dataset_from_directory(
    test_path,
    image_size = img_size,
    batch_size = batch_size,
    shuffle = FALSE
  )
  
  #Names of classes
  class_names <- test_ds_raw$class_names
  k <- length(class_names)
  
  # Predictions -> classes (1..k)
  pred_probs <- model %>% predict(test_ds_raw)
  y_pred <- max.col(pred_probs)  # 1..k
  
  # Get True labels (y_true) from the dataset
  y_true <- integer(0)
  it <- test_ds_raw$as_numpy_iterator()
  repeat {
    batch <- tryCatch(reticulate::iter_next(it), error = function(e) NULL)
    if (is.null(batch)) break
    y_true <- c(y_true, as.integer(batch[[2]]))
  }
  
  #Harmonize 0-based vs 1-based
  # y_pred is 1..k ; y_true is 0..k-1
  if (min(y_true) == 0) y_true <- y_true + 1
  
  ## To convert in label text
  ref <- factor(class_names[y_true], levels = class_names)  # True
  dat <- factor(class_names[y_pred], levels = class_names)  # Pred
  
  ###3.10.1 Confusion matrix and statistics
  
  t_conf <- caret::confusionMatrix(dat, ref)
  cat("=== Confusion Matrix and statitstics ===")
  print(t_conf)
  
  #Confusion matrix + accuracy
  cm <- t_conf$table
  
  write.csv(cm, paste0(out_dir, model_name, "_confusion.matrix.csv"))
  
  ###3.10.2 Metrics for the model (precision, recall, accuracy, F1 score)
  
  tp <- diag(cm)
  fp <- rowSums(cm) - tp
  fn <- colSums(cm) - tp
  
  precision <- tp / (tp + fp)
  recall <- tp / (tp + fn)
  
  f1 <- 2 * precision * recall / (precision + recall)
  
  macro_F1 <- mean(f1, na.rm = TRUE)
  macro_precision <- mean(precision, na.rm = TRUE)
  macro_recall <- mean(recall, na.rm = TRUE)
  macro_accuracy <- round((t_conf$overall[1] * 100), 2)
  
  
  # table by class: weighted F1
  f1_table <- data.frame(
    class = colnames(cm),
    precision = round(precision, 3),
    recall = round(recall, 3),
    F1 = round(f1, 3)
  )
  
  
  dataset_model <- data_frame(Model = model_name,
                              Accuracy = round((t_conf$overall[1] * 100), 2),
                              Accuracy_lower = round((t_conf$overall[3] * 100), 2),
                              Accuracy_Upper = round((t_conf$overall[4] * 100), 2),
                              Accuracy_null_model = round((t_conf$overall[5] * 100), 2),
                              Accuracy_pvalue = t_conf$overall[6],
                              Sensitivity = round((t_conf$byClass[1] * 100), 2),
                              Specificity = round((t_conf$byClass[2] * 100), 2),  # True Negative Rate
                              Precision = round((t_conf$byClass[3] * 100), 2),
                              Macro_F1=  round(mean(f1, na.rm = TRUE) * 100, 2),
                              Macro_recall = round(mean(recall, na.rm = TRUE)*100, 2))
  
  write.csv(dataset_model, paste0(out_dir, model_name, "_dataset_model.csv"))
  
  results <- list(metrics= dataset_model, F1byclass= f1_table, model_trained = model)
  
  
  #3.11. Figures 
  
  ###3.11.1. Plotting accuracy/loss evolution through epochs
  plot_history <- function(hist, title_prefix = "Training") {
    
    # df containing epochs, accuracy and loss from both first draft and fine-tuning 
    df <- data.frame(
      epoch = seq_along(hist$metrics$loss),
      train_acc = hist$metrics$accuracy,
      val_acc = hist$metrics$val_accuracy,
      train_loss = hist$metrics$loss,
      val_loss = hist$metrics$val_loss
    )
    
    # --- Accuracy ---
    acc_long <- df %>%
      dplyr::select(epoch, train_acc, val_acc) %>%
      pivot_longer(-epoch, names_to = "type", values_to = "value")
    
    p_acc <- ggplot(acc_long, aes(epoch, value, color = type)) +
      geom_line(linewidth = 1) +
      theme_minimal() +
      labs(
        title = paste(title_prefix, "Accuracy"),
        x = "Epoch",
        y = "Accuracy",
        color = ""
      )
    
    # --- Loss ---
    loss_long <- df %>%
      dplyr::select(epoch, train_loss, val_loss) %>%
      pivot_longer(-epoch, names_to = "type", values_to = "value")
    
    p_loss <- ggplot(loss_long, aes(epoch, value, color = type)) +
      geom_line(linewidth = 1) +
      theme_minimal() +
      labs(
        title = paste(title_prefix, "Loss"),
        x = "Epoch",
        y = "Loss",
        color = ""
      )
    
    #Combine both graphs:
    combined_plot <- p_acc / p_loss
    
    return(combined_plot)
  }
  
  p1 <- plot_history(history, "Initial Training")
  p1
  
  p2 <- plot_history(history_ft, "Fine-Tuning")
  p2
  
  #Saving the graphs for first draft (initial) and fine-tuning:
  ggsave(paste0(out_dir_figure,"/Fig_training_curves_initial2.pdf"), p1, width = 8, height = 10)
  ggsave(paste0(out_dir_figure,"/Fig_training_curves_finetune2.pdf"), p2, width = 8, height = 10)
  
  ###3.11.2. Heatmap of confusion matrix
  cm_prop <- prop.table(cm, margin = 2)  # normalization by column ("True" class) 
  cm_prop_df <- as.data.frame(cm_prop)
  
  p1 <- ggplot(cm_prop_df, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = sprintf("%.2f", Freq)), size = 3) +
    theme_minimal() +
    labs(
      title = "Confusion matrix (normalized by true class)",
      x = "True label",
      y = "Predicted label"
    )
  
  p1
  
  ggsave(paste0(out_dir_figure, model_name,
                "_Fig_confusion_matrix.pdf"),
         plot = p1,
         width = 8,
         height = 6
  )
  
  ###3.11.3. Unclassified images
  
  #get_ordered_files: renaming all images as their address inside the "test" directory
  get_ordered_files <- function(test_path, class_names) {
    files <- c()
    for (cl in class_names) {
      cl_dir <- fs::path(test_path, cl)
      # important: tri (pour coller à l'ordre du dataset)
      cl_files <- sort(fs::dir_ls(cl_dir, type = "file", glob = "*.jpg"))
      files <- c(files, cl_files)
    }
    files
  }
  
test_files <- get_ordered_files(test_path, class_names)
  
#Checking all files have been labeled succesfully by get_ordered_files
cat("nb files:", length(test_files), "\n")
cat("nb labels  :", length(y_true), "\n")
  
  #to_name: changes the numerical tags classes had in y_pred and y_true (as seen in point 5.1.) for their correspondent names from class_names
  k <- length(class_names)
  to_name <- function(y) {
    if (min(y) == 0) class_names[y + 1] else class_names[y]
  }
  
  true_names <- to_name(y_true)
  pred_names <- to_name(y_pred)
  
  #Search for misidentified leaves by checking where the y_pred and y_true tags don't match:
  wrong_idx <- which(true_names != pred_names)
  cat("Nb error:", length(wrong_idx), "\n")
  
  #Search which leaves have been identified correctly
  right_idx <- which(true_names == pred_names)
  
  
  #show_misclassified: returns n misclassified leaves from the ones recoverd by wrong_idx, resizes them, and tags them with both the predicted and true class
  show_misclassified <- function(n = wrong_idx, seed = 1, resize = "256x256") {
    set.seed(seed)
    idx <- sample(wrong_idx, min(n, length(wrong_idx)))
    
    imgs <- mapply(function(p, t, pr) {
      img <- image_read(p) |> image_resize(resize)
      image_annotate(
        img,
        text = paste0("True: ", t, "\nPred: ", pr),
        location = "+10+20",
        size = 20
      )
    }, test_files[idx], true_names[idx], pred_names[idx], SIMPLIFY = FALSE)
    
    image_montage(image_join(imgs), tile = paste0(ceiling(sqrt(length(imgs))), "x"))
  }
  
  #Show the misclassified images in a grid
  img_grid <- show_misclassified(n = wrong_idx, seed = 42)
  
  image_write(
    img_grid,
    path = paste0(out_dir_figure, model_name,
                  "_Fig_misclassified_examples.pdf"),
    format = "pdf"
  )
  
  image_write(
    img_grid,
    path = paste0(out_dir_figure, model_name,
                  "_Fig_misclassified_examples.png"),
    format = "png",
    density = 300
  )
  
  #show_confusion_pair: similar to show_misclassified, but showing all images missidentified the same way (they share both the true and the (wrongly) predicted class tag)
  show_confusion_pair <- function(true_class, pred_class, n = wrong_idx) {
    idx <- which(true_names == true_class & pred_names == pred_class)
    if (length(idx) == 0) { cat("Aucune image pour ce couple.\n"); return(invisible(NULL)) }
    idx <- idx[1:min(n, length(idx))]
    
    imgs <- lapply(idx, function(i) {
      img <- image_read(test_files[i]) |> image_resize("256x256")
      image_annotate(img, text = fs::path_file(test_files[i]), location = "+10+20", size = 16)
    })
    
    image_montage(image_join(imgs), tile = paste0(ceiling(sqrt(length(imgs))), "x"))
  }
  
  p3 <- show_confusion_pair("HEL", "HIB", n = 3)
  
  image_write(
    p3,
    path = paste0(out_dir_figure, model_name,
                  "_fig_confusion_hel_to_hib.pdf"),
    format = "pdf"
  )
  
  p3
  
  p4 <- show_confusion_pair("HIB", "HEL", n = 3)
  
  image_write(
    p4,
    path = paste0(out_dir_figure, model_name,
                  "_fig_confusion_hib_to_hel.pdf"),
    format = "pdf"
  )
  p4 
  
  return(results)
  
  ###3.11.4. Lists of classified/misclassified images
  test_files_nm <- str_remove(test_files, paste0(out_dir, "/test/([:alpha:]+)/"))
  
  list_right <- c()
  for (i in (right_idx)){
    list_right <- c(list_right, (test_files_nm[[i]]))
  }
  
  list_right <- as.data.frame.character(list_right)
  write.csv(list_right, paste0(out_dir, "correct_list"))
  
  list_wrong <- c()
  for (i in (wrong_idx)){
    list_wrong <- c(list_wrong, (test_files_nm[[i]]))
  }
  
  list_wrong <- as.data.frame.character(list_wrong)
  write.csv(list_wrong, paste0(out_dir, "wrong_list"))
  
  ###3.11.5. Prediction matrix
  test_files_nm_df <- as.data.frame(test_files_nm)
  pred_probs <- cbind(test_files_nm_df, pred_probs, row.names = NULL)
  colnames(pred_probs) <- c("leaf", class_names)
  write.csv(pred_probs, paste0(out_dir, "pred_probs"))
}

#5. NULL CNN MODEL (no data_augmentation)----
#Everything here is overwritten with the variations in 6.
CNN_model_no_data_augmentation <- function(model_name = "El_model",
                               src_dir = "data/leaf_masks_nuevas", #source directory with leaf masks
                               out_dir = paste0("data/", model_name, "/") , # output file with train/valid/test
                               out_dir_figure = paste0("data/", model_name, "/figures/") ,
                               p_train  = 0.8, 
                               p_valid  = 0.1,            # remaining = test
                               copy_files = TRUE,             # TRUE = copy, FALSE = move
                               target = "sp",              # "sp" ou "leaftype" ou "sp_leaftype"
                               batch_size = 32,
                               img_size = c(224, 224)
){
  
  ##3.1. Set up model parameters
  
  set.seed(30)
  
  #Building the out_dir (if necessary)
  # dir_delete(out_dir)
  dir_create(out_dir)
  # dir_delete(out_dir_figure)
  dir_create(out_dir_figure)
  
  ##3.2. Build metadata (FORMAT: AZO_01SRC25(1)_1_RP.jpg)
  
  #List all images in src_dir
  files_all <- dir_ls(src_dir, type = "file", glob = "*.jpg")
  
  #meta: data frame containing the images' paths + name decomposition (used in image sorting)
  meta <- tibble(path = files_all) %>%
    mutate(
      file = file_path_sans_ext(path_file(path)),
      ext  = file_ext(path),
      m = str_match(file, "^([A-Z]+)_([^()]+)\\((\\d+)\\)_(\\d+)_([A-Z]+)$")
    ) %>%
    mutate(
      sp        = m[,2],
      pop       = m[,3],
      ind       = as.integer(m[,4]),
      numbranch = as.integer(m[,5]),
      leaftype  = m[,6]
    ) %>%
    dplyr::select(-m) %>%
    filter(!is.na(sp), !is.na(leaftype))
  
  #Creating the "class_dir" variable inside the "meta" data frame, used in the next step
  meta <- meta %>%
    mutate(
      class = if (target == "sp") {
        sp
      } else if (target == "leaftype") {
        leaftype
      } else if (target == "sp_leaftype") {
        paste(sp, leaftype, sep = "__")
      } else {
        sp
      }
    )
  
  meta <- meta %>%
    mutate(
      class_dir = class %>%
        as.character() %>%
        stringr::str_to_lower() %>%
        stringr::str_replace_all("[^a-z0-9]+", "_") %>%
        stringr::str_replace_all("^_|_$", ""),
      gid = paste(sp, pop, ind, sep = "__")   
    )
  
  #3.3. Split images as training, test, validation
  
  #Here stratified (80/10/10 by class) with attributing all picture of one individual to the same dataset
  set.seed(30)
  
  #Table of individuals and their correspondent species:
  group_df <- meta %>%
    dplyr::distinct(gid, class_dir)
  
  #split_groups_one_class: each individual (and, in consequence, their leaves) is assigned one dataset partition 
  split_groups_one_class <- function(df, p_train, p_valid) {
    n <- nrow(df)
    idx <- sample.int(n)
    
    n_train <- floor(p_train * n)
    n_valid <- floor(p_valid * n)
    
    df %>%
      mutate(.split = dplyr::case_when(
        row_number() %in% idx[1:n_train] ~ "train",
        row_number() %in% idx[(n_train + 1):(n_train + n_valid)] ~ "valid",
        TRUE ~ "test"
      ))
  }
  
  group_split <- group_df %>%
    dplyr::group_by(class_dir) %>%
    dplyr::group_modify(~ split_groups_one_class(.x, p_train, p_valid)) %>%
    dplyr::ungroup()
  
  #Attaching the result of split_groups_one_class to the data frame "meta"
  meta_split <- meta %>%
    dplyr::left_join(group_split, by = c("gid", "class_dir"))
  
  cat("=== Dataset spliting ===\n")
  print(meta_split %>% count(.split, class_dir))
  
  #Assigning the images its correspondent dataset partition according to the .split column in meta_split (created with the split_groups_one_class function)
  splits <- c("train", "valid", "test")
  classes <- sort(unique(meta_split$class_dir))
  
  for (sp in splits) {
    for (cl in classes) {
      dir_create(path(out_dir, sp, cl), recurse = TRUE)
    }
  }
  
  #move_or_copy: function in charge of moving the images
  move_or_copy <- function(src, dest, do_copy = TRUE) {
    if (!file_exists(src)) return(FALSE)
    
    # avoid name collision
    dest_final <- dest
    if (file_exists(dest_final)) {
      ext <- path_ext(dest_final)
      base <- path_ext_remove(path_file(dest_final))
      parent <- path_dir(dest_final)
      i <- 1
      repeat {
        candidate <- path(parent, paste0(base, "_", i, ifelse(ext == "", "", paste0(".", ext))))
        if (!file_exists(candidate)) { dest_final <- candidate; break }
        i <- i + 1
      }
    }
    
    if (do_copy) file_copy(src, dest_final, overwrite = FALSE) else file_move(src, dest_final)
    TRUE
  }
  
  results <- meta_split %>%
    mutate(
      filename = path_file(path),
      dest = path(out_dir, .split, class_dir, filename),
      ok = mapply(move_or_copy, path, dest, MoreArgs = list(do_copy = copy_files))
    )
  
  #Checking all images are in their correspondent folders inside out_dir
  cat("Source files missing:", sum(!results$ok), "\n")
  print(results %>% count(.split, class_dir))
  
  #.csv file for meta_split (name of the leaves, their dataset partition and the check we run last step)
  write.csv(results, paste0(out_dir, model_name, "_dataset_CNN.csv"))
  
  #3.4. Keras datasets
  
  seed <- 123
  
  train_ds <- image_dataset_from_directory(
    path(out_dir, "train"),
    image_size = img_size,
    batch_size = batch_size,
    seed = seed
  )
  
  valid_ds <- image_dataset_from_directory(
    path(out_dir, "valid"),
    image_size = img_size,
    batch_size = batch_size,
    seed = seed
  )
  
  #Defining the name and quantity of our classes (used for class weights)
  class_names <- train_ds$class_names
  num_classes <- length(class_names)
  cat("=== Classes analyzed ===\n")
  print(class_names)
  
  #3.5. Shufflining and prefetching
  #Image shuffling allows more randomness when training the model.
  #Prefetching speeds up the process of training the model by preloading n number of images from which the model will use whatever number is batch_size in the next step of the training
  #We are going to let TensorFlow decide how many images to preload by using the AUTOTUNE feature
  
  AUTOTUNE <- tf$data$AUTOTUNE
  
  train_ds <- train_ds %>%
    tf$data$Dataset$shuffle(buffer_size = as.integer(1000L)) %>%
    tf$data$Dataset$prefetch(buffer_size = AUTOTUNE)
  
  valid_ds <- valid_ds %>%
    tf$data$Dataset$prefetch(buffer_size = AUTOTUNE)
  
  #3.6. Class weights (way to deal with image quantity/species desequilibrium)
  
  train_counts <- results %>%
    filter(.split == "train") %>%
    count(class_dir)
  
  #Inverse weight: more rare  => bigger weights
  w_by_classdir <- with(train_counts, setNames(max(n) / n, class_dir))
  
  #Mapper on the order of  class_names (order of the files)
  w_vec <- as.numeric(w_by_classdir[class_names])
  class_weight <- as.list(w_vec)
  
  names(class_weight) <- as.character(0:(num_classes - 1))
  
  cat("=== Weighted classification selected by the model (for unbalanced data) ===\n")
  print(class_weight)
  
  #3.8. Running the model
  #(MobileNetV2 + augmentation of the data) + FIT with class_weight
  cat("Model run  == MobileNetV2\n")
  base_model <- application_mobilenet_v2(
    input_shape = c(img_size, 3),
    include_top = FALSE,
    weights = "imagenet"
  )
  base_model$trainable <- FALSE
  
  inputs <- layer_input(shape = c(img_size, 3))
  
  #Adding a pixel color value reescaling and a global_average_pooling_2d layer: transforming our images into two-dimensional data structures (numerical values between 0 and 1, organized in rows and columns)
  #The "layer_dropout" layer helps with overfitting (adjusting our model too much to the "train" dataset, resulting in bad scores with the "validation" dataset) by dropping 20% of our training results.
  x <- inputs %>%
    #data_augmentation() %>%
    layer_rescaling(1/255) %>%
    base_model() %>%
    layer_global_average_pooling_2d() %>%
    layer_dropout(0.2)
  
  #Finally, we add a layer in charge of adjusting our results into a probability distribution
  outputs <- x %>% layer_dense(num_classes, activation = "softmax")
  model <- keras_model(inputs, outputs)
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = 1e-3),
    loss = "sparse_categorical_crossentropy",
    metrics = c("accuracy")
  )
  
  #The "callbacks" object is used to stop the training once the learning_rate drops below a certain value. In this model, it also reduces the learning rate to a 20% if the learning rate stays the same for 2 epochs.
  callbacks <- list(
    callback_early_stopping(patience = 5, restore_best_weights = TRUE),
    callback_reduce_lr_on_plateau(patience = 2, factor = 0.2)
  )
  
  history <- model %>% fit(
    train_ds,
    validation_data = valid_ds,
    epochs = 30,
    callbacks = callbacks,
    class_weight = class_weight
  )
  
  #3.9. Fine-tuning
  cat("=== Start fine-tuning ===\n")
  
  #Unlock backbone
  base_model$trainable <- TRUE
  
  #Recommended option: unlock only for the last layers
  for (layer in base_model$layers[1:(length(base_model$layers) - 30)]) {
    layer$trainable <- FALSE
  }
  
  #Recompiled with a smaller learning rate 
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = 1e-5),
    loss = "sparse_categorical_crossentropy",
    metrics = "accuracy"
  )
  
  #Resume learning 
  history_ft <- model %>% fit(
    train_ds,
    validation_data = valid_ds,
    epochs = 30,
    class_weight = class_weight,
    callbacks = callbacks
  )
  
  #3.10. Prediction for "test" dataset 
  
  test_path <- fs::path(out_dir, "test")
  stopifnot(fs::dir_exists(test_path))
  
  #Predictions
  test_ds_raw <- image_dataset_from_directory(
    test_path,
    image_size = img_size,
    batch_size = batch_size,
    shuffle = FALSE
  )
  
  #Names of classes
  class_names <- test_ds_raw$class_names
  k <- length(class_names)
  
  # Predictions -> classes (1..k)
  pred_probs <- model %>% predict(test_ds_raw)
  y_pred <- max.col(pred_probs)  # 1..k
  
  # Get True labels (y_true) from the dataset
  y_true <- integer(0)
  it <- test_ds_raw$as_numpy_iterator()
  repeat {
    batch <- tryCatch(reticulate::iter_next(it), error = function(e) NULL)
    if (is.null(batch)) break
    y_true <- c(y_true, as.integer(batch[[2]]))
  }
  
  #Harmonize 0-based vs 1-based
  # y_pred is 1..k ; y_true is 0..k-1
  if (min(y_true) == 0) y_true <- y_true + 1
  
  ## To convert in label text
  ref <- factor(class_names[y_true], levels = class_names)  # True
  dat <- factor(class_names[y_pred], levels = class_names)  # Pred
  
  ###3.10.1 Confusion matrix and statistics
  
  t_conf <- caret::confusionMatrix(dat, ref)
  cat("=== Confusion Matrix and statitstics ===")
  print(t_conf)
  
  #Confusion matrix + accuracy
  cm <- t_conf$table
  
  write.csv(cm, paste0(out_dir, model_name, "_confusion.matrix.csv"))
  
  ###3.10.2 Metrics for the model (precision, recall, accuracy, F1 score)
  
  tp <- diag(cm)
  fp <- rowSums(cm) - tp
  fn <- colSums(cm) - tp
  
  precision <- tp / (tp + fp)
  recall <- tp / (tp + fn)
  
  f1 <- 2 * precision * recall / (precision + recall)
  
  macro_F1 <- mean(f1, na.rm = TRUE)
  macro_precision <- mean(precision, na.rm = TRUE)
  macro_recall <- mean(recall, na.rm = TRUE)
  macro_accuracy <- round((t_conf$overall[1] * 100), 2)
  
  
  # table by class: weighted F1
  f1_table <- data.frame(
    class = colnames(cm),
    precision = round(precision, 3),
    recall = round(recall, 3),
    F1 = round(f1, 3)
  )
  
  
  dataset_model <- data_frame(Model = model_name,
                              Accuracy = round((t_conf$overall[1] * 100), 2),
                              Accuracy_lower = round((t_conf$overall[3] * 100), 2),
                              Accuracy_Upper = round((t_conf$overall[4] * 100), 2),
                              Accuracy_null_model = round((t_conf$overall[5] * 100), 2),
                              Accuracy_pvalue = t_conf$overall[6],
                              Sensitivity = round((t_conf$byClass[1] * 100), 2),
                              Specificity = round((t_conf$byClass[2] * 100), 2),  # True Negative Rate
                              Precision = round((t_conf$byClass[3] * 100), 2),
                              Macro_F1=  round(mean(f1, na.rm = TRUE) * 100, 2),
                              Macro_recall = round(mean(recall, na.rm = TRUE)*100, 2))
  
  write.csv(dataset_model, paste0(out_dir, model_name, "_dataset_model.csv"))
  
  results <- list(metrics= dataset_model, F1byclass= f1_table, model_trained = model)
  
  
  #3.11. Figures 
  
  ###3.11.1. Plotting accuracy/loss evolution through epochs
  plot_history <- function(hist, title_prefix = "Training") {
    
    # df containing epochs, accuracy and loss from both first draft and fine-tuning 
    df <- data.frame(
      epoch = seq_along(hist$metrics$loss),
      train_acc = hist$metrics$accuracy,
      val_acc = hist$metrics$val_accuracy,
      train_loss = hist$metrics$loss,
      val_loss = hist$metrics$val_loss
    )
    
    # --- Accuracy ---
    acc_long <- df %>%
      dplyr::select(epoch, train_acc, val_acc) %>%
      pivot_longer(-epoch, names_to = "type", values_to = "value")
    
    p_acc <- ggplot(acc_long, aes(epoch, value, color = type)) +
      geom_line(linewidth = 1) +
      theme_minimal() +
      labs(
        title = paste(title_prefix, "Accuracy"),
        x = "Epoch",
        y = "Accuracy",
        color = ""
      )
    
    # --- Loss ---
    loss_long <- df %>%
      dplyr::select(epoch, train_loss, val_loss) %>%
      pivot_longer(-epoch, names_to = "type", values_to = "value")
    
    p_loss <- ggplot(loss_long, aes(epoch, value, color = type)) +
      geom_line(linewidth = 1) +
      theme_minimal() +
      labs(
        title = paste(title_prefix, "Loss"),
        x = "Epoch",
        y = "Loss",
        color = ""
      )
    
    #Combine both graphs:
    combined_plot <- p_acc / p_loss
    
    return(combined_plot)
  }
  
  p1 <- plot_history(history, "Initial Training")
  p1
  
  p2 <- plot_history(history_ft, "Fine-Tuning")
  p2
  
  #Saving the graphs for first draft (initial) and fine-tuning:
  ggsave(paste0(out_dir_figure,"/Fig_training_curves_initial2.pdf"), p1, width = 8, height = 10)
  ggsave(paste0(out_dir_figure,"/Fig_training_curves_finetune2.pdf"), p2, width = 8, height = 10)
  
  ###3.11.2. Heatmap of confusion matrix
  cm_prop <- prop.table(cm, margin = 2)  # normalization by column ("True" class) 
  cm_prop_df <- as.data.frame(cm_prop)
  
  p1 <- ggplot(cm_prop_df, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = sprintf("%.2f", Freq)), size = 3) +
    theme_minimal() +
    labs(
      title = "Confusion matrix (normalized by true class)",
      x = "True label",
      y = "Predicted label"
    )
  
  p1
  
  ggsave(paste0(out_dir_figure, model_name,
                "_Fig_confusion_matrix.pdf"),
         plot = p1,
         width = 8,
         height = 6
  )
  
  ###3.11.3. Unclassified images
  
  #get_ordered_files: renaming all images as their address inside the "test" directory
  get_ordered_files <- function(test_path, class_names) {
    files <- c()
    for (cl in class_names) {
      cl_dir <- fs::path(test_path, cl)
      # important: tri (pour coller à l'ordre du dataset)
      cl_files <- sort(fs::dir_ls(cl_dir, type = "file", glob = "*.jpg"))
      files <- c(files, cl_files)
    }
    files
  }
  
  test_files <- get_ordered_files(test_path, class_names)
  
  #Checking all files have been labeled succesfully by get_ordered_files
  cat("nb files:", length(test_files), "\n")
  cat("nb labels  :", length(y_true), "\n")
  
  #to_name: changes the numerical tags classes had in y_pred and y_true (as seen in point 5.1.) for their correspondent names from class_names
  k <- length(class_names)
  to_name <- function(y) {
    if (min(y) == 0) class_names[y + 1] else class_names[y]
  }
  
  true_names <- to_name(y_true)
  pred_names <- to_name(y_pred)
  
  #Search for misidentified leaves by checking where the y_pred and y_true tags don't match:
  wrong_idx <- which(true_names != pred_names)
  cat("Nb error:", length(wrong_idx), "\n")
  
  #Search which leaves have been identified correctly
  right_idx <- which(true_names == pred_names)
  
  
  #show_misclassified: returns n misclassified leaves from the ones recoverd by wrong_idx, resizes them, and tags them with both the predicted and true class
  show_misclassified <- function(n = wrong_idx, seed = 1, resize = "256x256") {
    set.seed(seed)
    idx <- sample(wrong_idx, min(n, length(wrong_idx)))
    
    imgs <- mapply(function(p, t, pr) {
      img <- image_read(p) |> image_resize(resize)
      image_annotate(
        img,
        text = paste0("True: ", t, "\nPred: ", pr),
        location = "+10+20",
        size = 20
      )
    }, test_files[idx], true_names[idx], pred_names[idx], SIMPLIFY = FALSE)
    
    image_montage(image_join(imgs), tile = paste0(ceiling(sqrt(length(imgs))), "x"))
  }
  
  #Show the misclassified images in a grid
  img_grid <- show_misclassified(n = wrong_idx, seed = 42)
  
  image_write(
    img_grid,
    path = paste0(out_dir_figure, model_name,
                  "_Fig_misclassified_examples.pdf"),
    format = "pdf"
  )
  
  image_write(
    img_grid,
    path = paste0(out_dir_figure, model_name,
                  "_Fig_misclassified_examples.png"),
    format = "png",
    density = 300
  )
  
  #show_confusion_pair: similar to show_misclassified, but showing all images missidentified the same way (they share both the true and the (wrongly) predicted class tag)
  show_confusion_pair <- function(true_class, pred_class, n = wrong_idx) {
    idx <- which(true_names == true_class & pred_names == pred_class)
    if (length(idx) == 0) { cat("Aucune image pour ce couple.\n"); return(invisible(NULL)) }
    idx <- idx[1:min(n, length(idx))]
    
    imgs <- lapply(idx, function(i) {
      img <- image_read(test_files[i]) |> image_resize("256x256")
      image_annotate(img, text = fs::path_file(test_files[i]), location = "+10+20", size = 16)
    })
    
    image_montage(image_join(imgs), tile = paste0(ceiling(sqrt(length(imgs))), "x"))
  }
  
  p3 <- show_confusion_pair("HEL", "HIB", n = 3)
  
  image_write(
    p3,
    path = paste0(out_dir_figure, model_name,
                  "_fig_confusion_hel_to_hib.pdf"),
    format = "pdf"
  )
  
  p3
  
  p4 <- show_confusion_pair("HIB", "HEL", n = 3)
  
  image_write(
    p4,
    path = paste0(out_dir_figure, model_name,
                  "_fig_confusion_hib_to_hel.pdf"),
    format = "pdf"
  )
  p4 
  
  return(results)
  
  ###3.11.4. Lists of classified/misclassified images
  test_files_nm <- str_remove(test_files, paste0(out_dir, "/test/([:alpha:]+)/"))
  
  list_right <- c()
  for (i in (right_idx)){
    list_right <- c(list_right, (test_files_nm[[i]]))
  }
  
  list_right <- as.data.frame.character(list_right)
  write.csv(list_right, paste0(out_dir, "correct_list"))
  
  list_wrong <- c()
  for (i in (wrong_idx)){
    list_wrong <- c(list_wrong, (test_files_nm[[i]]))
  }
  
  list_wrong <- as.data.frame.character(list_wrong)
  write.csv(list_wrong, paste0(out_dir, "wrong_list"))
  
  ###3.11.5. Prediction matrix
  test_files_nm_df <- as.data.frame(test_files_nm)
  pred_probs <- cbind(test_files_nm_df, pred_probs, row.names = NULL)
  colnames(pred_probs) <- c("leaf", class_names)
  write.csv(pred_probs, paste0(out_dir, "pred_probs"))
}

#6. VARIATIONS----

##6.1. El model (E 0.05, F no, I 240, all remaining 0.1)----
model_name <- "El"
El <- CNN_model_no_flipping (model_name = "El",
                                src_dir = , #source directory with leaf masks
                                out_dir = paste0("data/", model_name, "/"),  # output file with train/valid/test
                                out_dir_figure = paste0("data/", model_name, "/figures/"),
                              p_train  = 0.8, 
                              p_valid  = 0.1,            # remaining = test
                              copy_files = TRUE,             # TRUE = copy, FALSE = move
                              target = "sp",              # "sp" ou "leaftype" ou "sp_leaftype"
                              batch_size = 32,
                              img_size = c(224, 224),
                              erasing = 0.05, 
                              zoom = 0.1,
                              rotation = 0.1,
                              translation = 0.1)

  
 ##6.2. Zl model (Z 0.05, F no, I 224, all remaining 0.1)----
model_name <- "Zl"
Zl <- CNN_model_no_flipping (model_name = "Zl",
                             src_dir = , #source directory with leaf masks
                             out_dir = paste0("data/", model_name, "/"),  # output file with train/valid/test
                             out_dir_figure = paste0("data/", model_name, "/figures/"),
                             p_train  = 0.8, 
                             p_valid  = 0.1,            # remaining = test
                             copy_files = TRUE,             # TRUE = copy, FALSE = move
                             target = "sp",              # "sp" ou "leaftype" ou "sp_leaftype"
                             batch_size = 32,
                             img_size = c(224, 224),
                             erasing = 0.1, 
                             zoom = 0.05,
                             rotation = 0.1,
                             translation = 0.1)

#Comparing the results:
table_compare_CNN <-rbind(El$metrics, Zl$metrics)

##6.3. Rl model (R 0.05, F no, I 224, all remaining 0.1)----
model_name <- "Rl" 
Rl <- CNN_model_no_flipping (model_name = "Rl",
                             src_dir = , #source directory with leaf masks
                             out_dir = paste0("data/", model_name, "/"),  # output file with train/valid/test
                             out_dir_figure = paste0("data/", model_name, "/figures/"),
                             p_train  = 0.8, 
                             p_valid  = 0.1,            # remaining = test
                             copy_files = TRUE,             # TRUE = copy, FALSE = move
                             target = "sp",              # "sp" ou "leaftype" ou "sp_leaftype"
                             batch_size = 32,
                             img_size = c(224, 224),
                             erasing = 0.1, 
                             zoom = 0.1,
                             rotation = 0.05,
                             translation = 0.1)
 
table_compare_CNN <-rbind(table_compare_CNN, Rl$metrics)

##6.4. Tl model (T 0.05, F no, I 224, all remaining 0.1)----
model_name <- "Tl" 
Tl <- CNN_model_no_flipping (model_name = "Tl",
                             src_dir = , #source directory with leaf masks
                             out_dir = paste0("data/", model_name, "/"),  # output file with train/valid/test
                             out_dir_figure = paste0("data/", model_name, "/figures/"),
                             p_train  = 0.8, 
                             p_valid  = 0.1,            # remaining = test
                             copy_files = TRUE,             # TRUE = copy, FALSE = move
                             target = "sp",              # "sp" ou "leaftype" ou "sp_leaftype"
                             batch_size = 32,
                             img_size = c(224, 224),
                             erasing = 0.1, 
                             zoom = 0.1,
                             rotation = 0.1,
                             translation = 0.05)

table_compare_CNN <- rbind(table_compare_CNN, Tl$metrics)

##6.5. Fn model (F no, I 224, all remaining 0.1)----
model_name <- "Fn" 
Fn <- CNN_model_no_flipping (model_name = "Fn",
                             src_dir = , #source directory with leaf masks
                             out_dir = paste0("data/", model_name, "/"),  # output file with train/valid/test
                             out_dir_figure = paste0("data/", model_name, "/figures/"),
                             p_train  = 0.8, 
                             p_valid  = 0.1,            # remaining = test
                             copy_files = TRUE,             # TRUE = copy, FALSE = move
                             target = "sp",              # "sp" ou "leaftype" ou "sp_leaftype"
                             batch_size = 32,
                             img_size = c(224, 224),
                             erasing = 0.1, 
                             zoom = 0.1,
                             rotation = 0.1,
                             translation = 0.1)

table_compare_CNN <-rbind(table_compare_CNN, Fn$metrics)

##6.6. Fy model (F yes, I 224, all remaining 0.1)----
model_name <- "Fy" 
Fy <- CNN_model_flipping (model_name = "Fy",
                             src_dir = , #source directory with leaf masks
                             out_dir = paste0("data/", model_name, "/"),  # output file with train/valid/test
                             out_dir_figure = paste0("data/", model_name, "/figures/"),
                             p_train  = 0.8, 
                             p_valid  = 0.1,            # remaining = test
                             copy_files = TRUE,             # TRUE = copy, FALSE = move
                             target = "sp",              # "sp" ou "leaftype" ou "sp_leaftype"
                             batch_size = 32,
                             img_size = c(224, 224),
                             erasing = 0.1, 
                             zoom = 0.1,
                             rotation = 0.1,
                             translation = 0.1)

table_compare_CNN <-rbind(table_compare_CNN, Fy$metrics)

##6.7. Il model (I 148, F no, all remaining 0.1)----
model_name <- "Il" 
Il <- CNN_model_no_flipping (model_name = "Il",
                          src_dir = , #source directory with leaf masks
                          out_dir = paste0("data/", model_name, "/"),  # output file with train/valid/test
                          out_dir_figure = paste0("data/", model_name, "/figures/"),
                          p_train  = 0.8, 
                          p_valid  = 0.1,            # remaining = test
                          copy_files = TRUE,             # TRUE = copy, FALSE = move
                          target = "sp",              # "sp" ou "leaftype" ou "sp_leaftype"
                          batch_size = 32,
                          img_size = c(148, 148),
                          erasing = 0.1, 
                          zoom = 0.1,
                          rotation = 0.1,
                          translation = 0.1)
table_compare_CNN <-rbind(table_compare_CNN, Il$metrics)

##6.8. Ih model (I 300, F no, all remaining 0.1)----
model_name <- "Ih" 
Ih <- CNN_model_no_flipping (model_name = "Ih",
                             src_dir = , #source directory with leaf masks
                             out_dir = paste0("data/", model_name, "/"),  # output file with train/valid/test
                             out_dir_figure = paste0("data/", model_name, "/figures/"),
                             p_train  = 0.8, 
                             p_valid  = 0.1,            # remaining = test
                             copy_files = TRUE,             # TRUE = copy, FALSE = move
                             target = "sp",              # "sp" ou "leaftype" ou "sp_leaftype"
                             batch_size = 32,
                             img_size = c(300, 300),
                             erasing = 0.1, 
                             zoom = 0.1,
                             rotation = 0.1,
                             translation = 0.1)
table_compare_CNN <-rbind(table_compare_CNN, Ih$metrics)

##6.9. Eh model (E 0.15, F no, I 224, all remaining 0.1)----
model_name <- "Eh" 
Eh <- CNN_model_no_flipping (model_name = "Eh",
                             src_dir = , #source directory with leaf masks
                             out_dir = paste0("data/", model_name, "/"),  # output file with train/valid/test
                             out_dir_figure = paste0("data/", model_name, "/figures/"),
                             p_train  = 0.8, 
                             p_valid  = 0.1,            # remaining = test
                             copy_files = TRUE,             # TRUE = copy, FALSE = move
                             target = "sp",              # "sp" ou "leaftype" ou "sp_leaftype"
                             batch_size = 32,
                             img_size = c(224, 224),
                             erasing = 0.15, 
                             zoom = 0.1,
                             rotation = 0.1,
                             translation = 0.1)
table_compare_CNN <-rbind(table_compare_CNN, Eh$metrics)

##6.10. Zh model (Z 0.15, F no, I 224, all remaining 0.1)----
model_name <- "Zh" 
Zh <- CNN_model_no_flipping (model_name = "Zh",
                             src_dir = , #source directory with leaf masks
                             out_dir = paste0("data/", model_name, "/"),  # output file with train/valid/test
                             out_dir_figure = paste0("data/", model_name, "/figures/"),
                             p_train  = 0.8, 
                             p_valid  = 0.1,            # remaining = test
                             copy_files = TRUE,             # TRUE = copy, FALSE = move
                             target = "sp",              # "sp" ou "leaftype" ou "sp_leaftype"
                             batch_size = 32,
                             img_size = c(224, 224),
                             erasing = 0.1, 
                             zoom = 0.15,
                             rotation = 0.1,
                             translation = 0.1)
table_compare_CNN <-rbind(table_compare_CNN, Zh$metrics)

##6.11. Rh model (R 0.15, F no, all remaining 0.1)----
model_name <- "Rh" 
Rh <- CNN_model_no_flipping (model_name = "Rh",
                             src_dir = , #source directory with leaf masks
                             out_dir = paste0("data/", model_name, "/"),  # output file with train/valid/test
                             out_dir_figure = paste0("data/", model_name, "/figures/"),
                             p_train  = 0.8, 
                             p_valid  = 0.1,            # remaining = test
                             copy_files = TRUE,             # TRUE = copy, FALSE = move
                             target = "sp",              # "sp" ou "leaftype" ou "sp_leaftype"
                             batch_size = 32,
                             img_size = c(224, 224),
                             erasing = 0.1, 
                             zoom = 0.1,
                             rotation = 0.15,
                             translation = 0.1)
table_compare_CNN <-rbind(table_compare_CNN, Rh$metrics)

##6.12. Th model (T 0.15, F no , all remaining 0.1)----
model_name <- "Th" 
Th <- CNN_model_no_flipping (model_name = "Th",
                             src_dir = , #source directory with leaf masks
                             out_dir = paste0("data/", model_name, "/"),  # output file with train/valid/test
                             out_dir_figure = paste0("data/", model_name, "/figures/"),
                             p_train  = 0.8, 
                             p_valid  = 0.1,            # remaining = test
                             copy_files = TRUE,             # TRUE = copy, FALSE = move
                             target = "sp",              # "sp" ou "leaftype" ou "sp_leaftype"
                             batch_size = 32,
                             img_size = c(224, 224),
                             erasing = 0.1, 
                             zoom = 0.1,
                             rotation = 0.1,
                             translation = 0.15)
table_compare_CNN <-rbind(table_compare_CNN, Th$metrics)
 
##6.13. Null model (I 224, no data_augmentation)----
model_name <- "Nd"
Nd <- CNN_model_no_data_augmentation (model_name = "Nd",
                                                   src_dir =  ,#source directory with leaf masks
                                                   out_dir = paste0("data/", model_name, "/") , # output file with train/valid/test
                                                   out_dir_figure = paste0("data/", model_name, "/figures/") ,
                                                   p_train  = 0.8, 
                                                   p_valid  = 0.1,            # remaining = test
                                                   copy_files = TRUE,             # TRUE = copy, FALSE = move
                                                   target = "sp",              # "sp" ou "leaftype" ou "sp_leaftype"
                                                   batch_size = 32,
                                                   img_size = c(224, 224)
                                                   
)
  table_compare_CNN <-rbind(table_compare_CNN, Nd$metrics)
