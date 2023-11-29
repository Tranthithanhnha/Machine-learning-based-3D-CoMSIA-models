# Machine-learning-based-3D-CoMSIA-models
Integration of machine learning in 3D-QSAR CoMSIA models for the identification of lipid antioxidant peptides
 The first group of models was built after the data pre-processing step using 24 estimators with default hyperparameters and without any feature selection (Script S1†). The second group of models was constructed using one of the aforementioned feature selection methods (Script S2.1–2.2†). In the first two groups, several parameters, including RCV2, Root Mean Squared Error (RMSE), Std_RMSE, and R2 (coefficient of determination for the training set), were computed to compare the performance and generalization of the models. R2_test (coefficient of determination for the test set) was used to assess their predictability.

Finally, several models with the best cross-validation (CV) statistics from the second group were selected for hyperparameter tuning, leading to the creation of the third group of models (Script S3.1–3.4†). Grid search and random search techniques were employed, along with five-fold cross-validation, to identify the optimal hyperparameters for the models. These models were trained and evaluated on the inner folds of the training set using different hyperparameter combinations, and the best hyperparameters were chosen based on their CV performance. GridSearch_CV has also been used to derive the optimum number of PLS components using each phase of feature selection (Script S4.1–4.4†).

RFE selection methods utilizing different estimators including RandomForest, XGBoost, and AdaBoost (Embedded-RFE of RF, XGB, Ada), were executed to assess the impact of these selection methods on the overall model performance (Script S5.1–5.3†). Bootstrapping (Rbstr2) and scrambling (p value) evaluation were also performed to assess the robustness and reliability of the ML-based models (Script S6.1–6.2†).
