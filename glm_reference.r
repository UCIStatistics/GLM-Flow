## iris is builtin to R, but we write to a CSV
## so that python can get the same data.
## the only change to the original dataset is to 
## replace the Species factor with an indicator for setosa.
iris_binary <- iris
iris_binary$Species <- as.numeric(iris_binary$Species == "versicolor")
write.csv(iris_binary, "iris.csv", row.names = FALSE)
iris_df <- read.csv("iris.csv") # this is exactly what python will read

## it would be easy to include the intercept as well.
glm(Species ~ . -1 , data=iris_df, family = "binomial")
