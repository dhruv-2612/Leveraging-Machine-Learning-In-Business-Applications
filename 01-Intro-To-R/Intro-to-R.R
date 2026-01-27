############################################################
# Introduction to R Programming
# This script demonstrates basic R concepts including:
# arithmetic, data structures, visualization, and logic
############################################################


############################################################
# 1. Using Built-in Data
############################################################

# The lynx dataset contains annual counts of lynx trappings
# We visualize it using a histogram
hist(lynx,
     main = "Histogram of Lynx Trappings",
     col = "lightblue",
     xlab = "Number of Lynx")


############################################################
# 2. Basic Arithmetic in R
############################################################

12 + 3
(12 / 3) * 3
(45 / 5) * 12 + 10
(23 * 3) - (43 * 2)


############################################################
# 3. Assignment Operator
############################################################

# Assigning values to variables using <-
X <- (45 / 5) * 12 + 10
Y <- (23 * 3) - (43 * 2)

# Subtracting values
X - Y


############################################################
# 4. Variables and Vectors
############################################################

# Numeric vector
PARatings <- c(3, 2, 4, 3, 5, 4, 3, 2, 3, 4, 5)

# Character vector
Name <- c("Ravi", "Sudha", "Ramesh", "Soham", "Shilpa",
          "Bhavesh", "Amarpreet", "Wilfred", "Abdul", "Srini", "Taposh")

# Numeric vector
Age <- c(23, 22, 23, 25, 26, 24, 27, 23, 23, 25, 26)


############################################################
# 5. Data Frames
############################################################

# Creating a data frame
PerfData <- data.frame(Name, PARatings, Age)

# Viewing the data frame
PerfData

# Inspecting the data
dim(PerfData)      # Dimensions
str(PerfData)      # Structure
summary(PerfData)  # Summary statistics


############################################################
# 6. Working with Files (Reproducible Method)
############################################################

# Writing data to a CSV file
write.csv(PerfData, "Appraisal.csv", row.names = FALSE)

# Reading data back from the CSV file
PerfData1 <- read.csv("Appraisal.csv")

# Inspecting imported data
str(PerfData1)
summary(PerfData1)


############################################################
# 7. Built-in Dataset: Iris
############################################################

# Loading the iris dataset
data(iris)

# Inspecting iris data
str(iris)
summary(iris)
head(iris)
tail(iris)


############################################################
# 8. Visualizing Categorical Data
############################################################

# Creating a frequency table
tab <- table(iris$Species)

# Bar plot
barplot(tab,
        main = "Species Distribution",
        xlab = "Species",
        ylab = "Frequency",
        col = rainbow(3))

# Pie chart
pie(tab,
    main = "Species Composition",
    col = rainbow(3))


############################################################
# 9. Visualizing Numerical Data
############################################################

# Histogram of Sepal Length
hist(iris$Sepal.Length,
     main = "Histogram of Sepal Length",
     xlab = "Sepal Length",
     col = "orange",
     breaks = 15)

# Scatter plot of Sepal Length vs Sepal Width
plot(iris$Sepal.Length, iris$Sepal.Width,
     col = "red",
     main = "Sepal Length vs Sepal Width",
     xlab = "Sepal Length",
     ylab = "Sepal Width")

# Multiple scatter plots
pairs(iris)

# Correlation plots using psych package
library(psych)
pairs.panels(iris)


############################################################
# 10. Data Types in R
############################################################

A <- 4
class(A)

B <- "gender"
class(B)

C <- TRUE
class(C)


############################################################
# 11. Vectors, Coercion, and Lists
############################################################

# Named vector
Age_named <- c(Ravi = 23, Sudha = 22, Ramesh = 23, Soham = 25, Shilpa = 26)
Age_named

# Type coercion in vectors
mixed <- c(23, 22, "Ravi", 25)
class(mixed)

# Lists can store mixed data types
mylist <- list(23, 22, "Ravi", TRUE)
sapply(mylist, class)


############################################################
# 12. Factors (Categorical Variables)
############################################################

PARatings_factor <- factor(PARatings)

PARatings_ordered <- factor(PARatings,
                            ordered = TRUE,
                            levels = c(2, 3, 4, 5))

str(PARatings_ordered)


############################################################
# 13. Conditional Statements
############################################################

x <- 10

if (x > 0) {
  "x is positive"
} else {
  "x is negative"
}

if (iris$Sepal.Length[1] < 4) {
  "Sepal length is less than 4"
} else {
  "Sepal length is greater than or equal to 4"
}


############################################################
# 14. Operators in R
############################################################

# Arithmetic operators
A <- 12
B <- 10

A * B
A - B

# Relational operators
A < B
A > B
A == B
A != B

# Logical operators
p <- 10

(p < 15 & p > 8)   # AND
(p > 15 | p > 8)   # OR
!(p < 15)          # NOT


############################################################
# End of Script
############################################################
