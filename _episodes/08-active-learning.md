---
title:  Active Learning 
teaching: 1
exercises: 0
questions:
- ""
objectives:
- ""
---

<!-- MathJax -->

<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>


# Active Learning for materials


Active learning is a data acquisition approach where the algorithm actively collaborates with a user to gather additional labeled data, improving its accuracy over time. Initially, the algorithm is trained on a small set of labeled examples and then identifies which new data points would be most beneficial to label. It queries the user to label these uncertain or informative samples, using the newly labeled data to enhance its performance. This cycle continues until the model reaches a desired level of accuracy.

**Example:** Consider training a machine learning model to recognize handwritten digits. The process begins with a limited set of labeled digits. The algorithm then selects digits it finds uncertain and asks the user to label them. By incorporating these newly labeled samples, the model refines its predictions, repeating this interactive labeling until it can reliably identify most handwritten digits.
