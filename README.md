# onequantum-mentorship-Apr2022
This repository would include the coding tasks assigned by my mentor, Thomas Skripka. <br>
# Creating a python package
The first task was to create a python project to solve a problem. He also provided me with those tutorials about the needed steps to create a python package:
https://packaging.python.org/en/latest/tutorials/packaging-projects/ <br>
https://docs.python.org/3/distutils/configfile.html
# The Closure Project
As I didn't have the chance to complete all the assigned tasks of the QNLP challenge at the Womanium 2022 Hackathon, we decided to work on our solution to fix it. By applying the python testing tools I've learnt from Tom during the program, I was able to debug and fix my solution as follows: <br>
The QNLP model failed to converge as the loss values accidently evaluated to nan, then the parameters were not updated and the model return rubbish values <br>
 After some investigations into the cause of the NAN values, I was working on solving the following issues: <br>
1- Pandas' data frames sometimes have NAN values for empty fields, I have used the following code to drop them if there are any: <br> df.dropna() <br>
2-The loss evaluation function has a division that might lead to NAN if the denominator evaluates to zero. The denominator was the size of the dataset, so I inserted an assert expression to make sure that the dataset is always non-empty. <br>
3- np.log was another suspicious source of NAN values if the argument value was too small, so I defined an epsilon variable that holds a very small float value and added it to the log argument. I learned this trick from the hackathon mentors. <br>
4- The last defense against NAN values was to use the np.nansum instead of the regular np.sum so that any remaining nan values would be added as zero. (This was an effective solution). But it might be destructive if it covers up other bugs. I would love to know your thoughts about that. <br>
The model training is now ending smoothly with no NAN values.
The next step is to fine-tune the model performance as the current accuracy is 84%. I would be satisfied if we crossed the 90% accuracy mark.
