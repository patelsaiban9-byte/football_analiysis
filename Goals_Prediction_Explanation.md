# How Goals Prediction Works (Simple)

This is a simple explanation of how the player goal prediction model works.

## 1. Inputs Used

The model takes these inputs:
- Age
- Position (FW/MF/DF/GK)
- Matches played
- Minutes played
- Assists
- xAG (Expected Assists)
- Shots

## 2. Data Preparation

Before training:
- Missing numeric values are filled using median.
- Missing position values are filled using mode.
- Position is converted to numbers:
  - FW = 3
  - MF = 2
  - DF = 1
  - GK = 0

## 3. Model Training

- A **Linear Regression** model is trained.
- Target column is **Goals**.
- Features are: Age, Position_Code, Matches, Minutes, Assists, xAG, Shots.

## 4. Prediction Step

When user enters values:
1. Inputs are converted into a single-row dataframe.
2. Model predicts goals.
3. If predicted value is negative, it is converted to 0.

## 5. Output

Final output shown:

`Predicted goals is <value>`

## 6. Formula (Simple)

Linear regression internally follows:

`Predicted Goals = b0 + b1*Age + b2*Position + b3*Matches + b4*Minutes + b5*Assists + b6*xAG + b7*Shots`

## 7. Note

If the dataset shows stronger relation between Shots and Goals, then Shots will naturally influence the prediction more than some other inputs.

## 8. Conclusion

This is a minor-project level prediction model that demonstrates:
- preprocessing
- training a regression model
- generating goal predictions from user inputs
