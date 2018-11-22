# NN-Based
## Dataset
[http://mulan.sourceforge.net/datasets-mlc.html][1]

### yeast
|name | domain | instances |nominal	|numeric|labels|cardinality	|density|distinct|
| ------ | ------ | ------ |------ |------ |------ |------ |------ |------ |
| yeast| biology | 2417	 |0|103	|14|4.237|0.303	|198|

## Evaluation
|evaluation criterion |BR | CC|ECC|
| ------ | ------ | -----|---|
| hamming loss|  0.2268266085059978 | 0.2268266085059978 | |
| ranking loss|  0.16849462724050177 |0.16860606695590194  | |
| one error| 0.24532453245324531 | 0.25192519251925194| |

## Requrements
- Python 3.6
- numpy 1.13.3
- scikit-learn 0.19.1


## Reference
[Jesse Read·Bernhard Pfahringer·Geoff Holmes·Eibe Frank, “Classifier chains for multi-label classification,” Machine Learning, vol. 85, no. 3, pp. 333–359, 2011][2]


  [1]: http://mulan.sourceforge.net/datasets-mlc.html
  [2]: https://link.springer.com/article/10.1007/s10994-011-5256-5





