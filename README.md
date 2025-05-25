# obligatorio-taller-ia

## Create environment:

* conda env create -f ./environment.yml \
* conda activate obl_taller_ia


### **Arquitectura de la Red**

* **Entrada** : 4 frames consecutivos (grises) de 84×84 píxeles apilados → tensor de 84×84×4.
* **Capa Conv 1** : 16 filtros de 8×8 con stride 4 + ReLU.
* **Capa Conv 2** : 32 filtros de 4×4 con stride 2 + ReLU.
* **Capa densa** : 256 neuronas + ReLU.
* **Salida** : una neurona por acción posible (valor Q estimado).
