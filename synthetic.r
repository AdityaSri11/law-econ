data2_base <- read.csv2(
  "data/data2.csv",
  sep = ";",
  dec = ".",
  stringsAsFactors = FALSE
)

# Convert all year columns to numeric by removing the comma thousand separator
cols_to_clean <- 2:ncol(data2_base)
data2_base[, cols_to_clean] <- lapply(data2_base[, cols_to_clean], function(x) {
  as.numeric(gsub(",", "", x))
})

# Inspect the loaded data
print(head(data2_base))
print(str(data2_base))
```