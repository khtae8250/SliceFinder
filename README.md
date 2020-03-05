# SliceFinder: Automated Data Slicing for Model Validation

We study the problem of automatic model analysis by finding interpretable, problematic, and large slices where the model underperforms. For example, an ML user may want to know that a trained model has a high loss for the demographic gender = Female & age in [30, 40] compared to other demographics.

The motivation of this paper is to complement existing approaches like TensorFlow Model Analysis (TFMA) where data slicing is done manually. While TFMA can be used by ML experts, it may be difficult for non-experts that do not have a good understanding of their data. We suspect that most Google Cloud users are non-experts that need assistance in model analysis. SliceFinder can automatically find slices of interest.

A slice is defined as a conjunction of feature-value pairs, which makes it interpretable. A slice is problematic if its loss is significantly higher than other data. Our solution is to treat each slice as a hypothesis and perform two tests: determine if the difference in loss is statistically significant and if the effect size of the difference is large enough. SliceFinder searches for slices efficiently by either traversing a decision tree or a lattice structure of slices. Since SliceFinder can only see a subset of slices, it uses alpha-investing to perform false discovery control. SliceFinder also favors larger slices when it can. The SliceFinder GUI allows users to quickly browse through problematic slices.

![customers1](https://user-images.githubusercontent.com/29707304/75939970-1f06c500-5ecf-11ea-97c5-b5a618def0d2.png)
![customers2](https://user-images.githubusercontent.com/29707304/75939980-275f0000-5ecf-11ea-8a9e-262002ab0f27.png)
  
A set of slices for customers in different regions. The naive approach of blindly collecting any data is not optimal. In the left figure, there may be enough American customer data, so collecting more data in this region may worsen the bias. Instead, as in the right figure, we would like to collect possibly-different amounts of data for slices to optimize both the model accuracy and fairness on all slices
