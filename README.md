# BinaryRelevance Based
## Dataset
[http://mulan.sourceforge.net/datasets-mlc.html][1]

### yeast
|name | domain | instances |nominal	|numeric|labels|cardinality	|density|distinct|
| ------ | ------ | ------ |------ |------ |------ |------ |------ |------ |
| yeast| biology | 2417	 |0|103	|14|4.237|0.303	|198|

## Evaluation
|evaluation criterion |BR | CC|ECC|PCC(效果很差)|
|---|---|---|---|---|
|hamming loss|0.2268266085059978|0.2268266085059978|0.23298021498675806 |0.5221218258295685|
|ranking loss|0.16849462724050177|0.16860606695590194|0.045019261908574866 ||
|one error| 0.24532453245324531|0.25192519251925194|0.24972737186477645||

## Requrements
- Python 3.6
- numpy 1.13.3
- scikit-learn 0.19.1

## Parameter
- ECC algorithm chain number:10
- ECC algorithm subset proportion:0.75

## Reference
[Jesse Read·Bernhard Pfahringer·Geoff Holmes·Eibe Frank, “Classifier chains for multi-label classification,” Machine Learning, vol. 85, no. 3, pp. 333–359, 2011][2]

[K. Dembczy´nski, W. Cheng, and E. H¨ullermeier, “Bayes optimal multilabel classification via probabilistic classifier chains,” in Proceedings of the 27th International Conference on Machine Learning, Haifa, Israel, 2010, pp. 279–286][3]

  [1]: http://mulan.sourceforge.net/datasets-mlc.html
  [2]: https://link.springer.com/article/10.1007/s10994-011-5256-5
  [3]: https://weiweicheng.com/research/papers/cheng-icml10c.pdf




