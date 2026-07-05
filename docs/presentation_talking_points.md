# Presentation Talking Points

## 1. Opening

This project is connected to my Career Experience Practicum course with IHI. In that course, I worked on bridge and infrastructure related business thinking. I wanted to turn that experience into a data automation prototype.

## 2. Problem

Infrastructure inspection data is large and updated every year. If analysts compare every bridge record manually, the process can be slow and inconsistent.

## 3. Automation idea

The prototype automates the first screening step. It reads historical bridge data, checks whether ratings decreased in the next year, trains a model, and creates a ranked priority list.

## 4. Why machine learning

The model can combine many signals at once, such as bridge age, traffic, structure length, condition ratings, load rating, and bridge type.

## 5. Why temporal validation

I do not randomly mix all years. I train on older years and test on the latest transition. This better represents real forecasting.

## 6. Result

The model is not perfect, but the top priority group has a higher concentration of actual deterioration cases than random selection. That means the model can support prioritization.

## 7. Safety statement

This is not a tool for deciding whether a bridge is safe. It only helps engineers decide which records to review first.

## 8. Business value for IHI style workflow

The value is human in the loop automation. The system reduces manual screening effort, while engineers still make final decisions.
