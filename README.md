# DeepQuality repository
## References
* (c) UPM 2024
* Joaquín Ordieres January 2024
* Part of DeepQuality project (https://deepquality.eu)

## Acknowledgement
The research described in the present deliverable has been developed within the project entitled “Use of robust deep learning methods for the automatic quality assessment of steel products (DeepQuality)” (G.A. No 101034037), which was funded by Research Fund for Coal and Steel (RFCS) of the European Union (EU). The sole responsibility of the issues treated in the present report lies with the authors; the Commission is not responsible for any use that may be made of the information contained therein. We wish to acknowledge with thanks the EU for the opportunity granted that has made possible the development of the present work.

## Structure
* CoilDetais class:  **/CoilDetais**
* NiFi repository of workflows: **/NiFi**
* Python scripts:  **\*.py**
* Bash scripts:    **\*.sh**

## Description
The chosen use case aims to investigate the development of a classification algorithm for analyzing 1-D data sets collectively to provide insights into product quality. Specifically, the industrial scenario under consideration involves evaluating the thickness of zinc coatings applied to hot dip galvanized steel coils, crucial for various industries including automotive.

In the hot dip galvanizing (HDG) process, thickness measurements are conducted using nine X-ray gauges, each assessing the zinc coating thickness on different faces of the coil. The quality of the galvanized steel strip depends on several factors, such as surface condition, zinc adhesion, and coating uniformity. While the first two factors are affected by substrate preparation and temperature control, evaluating coating uniformity often happens too late in the process for timely adjustments.

In practice, many manufacturers assess the zinc coating weight after the coil has left the mill. If the results are unsatisfactory, the entire coil may need to be downgraded or scrapped, leading to production delays, raw material wastage, and increased costs.

Once the CNN model technology was validated as a suitable solution for evaluating coils with human-adapted criteria, albeit reliant on the training process and a limited data sample, the next step is to deploy it in an unattended configuration capable of delivering robust assessments. More importantly, this deployment must also discern when the model becomes biased, signaling the need for retraining.

The framework revolves around a design phase conducted in the back-office, preceding the system's implementation. Subsequently, the implementation phase entails constructing an ensemble of selected models and utilizing them for predictions. Additionally, it involves integrating operator decisions to gauge the performance of individual models and determine their retraining requirements.

Practical implementation of this framework involves leveraging NiFi™ technology, chosen for its compatibility with modern microservice-oriented computing platforms orchestrated by Kubernetes. This NiFi™ technology operates within a testbed infrastructure driven by Kubernetes' microservice orchestrator, providing a sample context for deployment.

