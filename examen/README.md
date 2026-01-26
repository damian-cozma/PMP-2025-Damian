**Am adăugat acest README ulterior pentru a include output-urile (imagini/ploturi) generate de codul Python. La trimiterea inițială a examenului pe GitHub (trimis la timp) am omis aceste fișiere, presupunând că pot fi regenerate direct din cod. Codul nu a fost modificat; este vorba doar de output-uri generate automat pentru claritate.**

<img width="1000" height="1000" alt="image" src="https://github.com/user-attachments/assets/98180bf2-916d-4f37-8354-4dced65b2665" />

<img width="720" height="480" alt="image" src="https://github.com/user-attachments/assets/63f5c370-d817-4716-9d2e-d63184536d57" />

<img width="1200" height="600" alt="image" src="https://github.com/user-attachments/assets/18e16bed-a3f1-4f34-9898-1e6b0dc56c31" />

<img width="1200" height="600" alt="image" src="https://github.com/user-attachments/assets/97455284-49a6-423f-a49c-b19a26d50d4c" />

```python
  rentals     temp_c  humidity   wind_kph  is_holiday  season
0      657   7.783223  0.225520  18.057709           0  winter
1      632  16.691937  0.533523  15.975831           0  spring
2      546  17.044212  0.879505  13.235575           1  spring
3      650  18.460255  0.623327  18.159102           1  autumn
4      730  19.841524  0.244849   0.500000           1  summer

missing values:
 rentals       0
temp_c        0
humidity      0
wind_kph      0
is_holiday    0
season        0
dtype: int64

season counts:
 season
summer    142
autumn    129
spring    120
winter    109
Name: count, dtype: int64

Initializing NUTS using jitter+adapt_diag...
Multiprocess sampling (4 chains in 4 jobs)
NUTS: [alpha, beta, sigma]
                                                                                
                              Step      Grad      Sampli…                       
  Progre…   Draws   Diverg…   size      evals     Speed     Elapsed   Remaini…  
 ────────────────────────────────────────────────────────────────────────────── 
  ━━━━━━━   3000    0         0.39      15        1829.40   0:00:01   0:00:00   
                                                  draws/s                       
  ━━━━━━━   3000    0         0.41      7         1791.86   0:00:01   0:00:00   
                                                  draws/s                       
  ━━━━━━━   3000    0         0.37      15        1739.27   0:00:01   0:00:00   
                                                  draws/s                       
  ━━━━━━━   3000    0         0.34      7         1665.52   0:00:01   0:00:00   
                                                  draws/s                       
                                                                                
Sampling 4 chains for 1_500 tune and 1_500 draw iterations (6_000 + 6_000 draws total) took 2 seconds.
                      mean     sd  hdi_5%  ...  ess_bulk  ess_tail  r_hat
alpha                0.340  0.043   0.271  ...    4343.0    4624.0    1.0
beta[temp_s]         0.519  0.038   0.458  ...    4964.0    3969.0    1.0
beta[hum_s]         -0.258  0.020  -0.289  ...    6127.0    4438.0    1.0
beta[wind_s]        -0.332  0.020  -0.364  ...    6941.0    4137.0    1.0
beta[is_holiday]    -0.369  0.040  -0.436  ...    6833.0    4125.0    1.0
beta[season_spring] -0.062  0.056  -0.148  ...    5409.0    5010.0    1.0
beta[season_summer]  0.017  0.067  -0.094  ...    4964.0    4051.0    1.0
beta[season_winter] -0.691  0.076  -0.816  ...    5191.0    4738.0    1.0
sigma                0.447  0.014   0.423  ...    6833.0    4502.0    1.0

[9 rows x 9 columns]
Initializing NUTS using jitter+adapt_diag...
Multiprocess sampling (4 chains in 4 jobs)
NUTS: [alpha, beta, sigma]
                                                                                
                              Step      Grad      Sampli…                       
  Progre…   Draws   Diverg…   size      evals     Speed     Elapsed   Remaini…  
 ────────────────────────────────────────────────────────────────────────────── 
  ━━━━━━━   3000    0         0.28      15        1747.75   0:00:01   0:00:00   
                                                  draws/s                       
  ━━━━━━━   3000    0         0.32      7         1773.11   0:00:01   0:00:00   
                                                  draws/s                       
  ━━━━━━━   3000    0         0.28      15        1709.08   0:00:01   0:00:00   
                                                  draws/s                       
  ━━━━━━━   3000    0         0.33      15        1655.10   0:00:01   0:00:00   
                                                  draws/s                       
                                                                                
Sampling 4 chains for 1_500 tune and 1_500 draw iterations (6_000 + 6_000 draws total) took 2 seconds.

Poly
                      mean     sd  hdi_2.5%  ...  ess_bulk  ess_tail  r_hat
alpha                0.386  0.043     0.302  ...    5571.0    4529.0    1.0
beta[temp_s]         0.511  0.036     0.440  ...    6205.0    4222.0    1.0
beta[temp2_s]       -0.152  0.020    -0.192  ...    8089.0    4547.0    1.0
beta[hum_s]         -0.267  0.019    -0.303  ...    9361.0    4216.0    1.0
beta[wind_s]        -0.324  0.019    -0.362  ...    9967.0    4333.0    1.0
beta[is_holiday]    -0.367  0.038    -0.440  ...    7841.0    4686.0    1.0
beta[season_spring] -0.060  0.054    -0.168  ...    6063.0    5237.0    1.0
beta[season_summer]  0.206  0.068     0.069  ...    7023.0    4375.0    1.0
beta[season_winter] -0.460  0.080    -0.620  ...    5753.0    4571.0    1.0
sigma                0.423  0.014     0.398  ...    8702.0    4310.0    1.0

[10 rows x 9 columns]

Top beta (linear): [('season_winter', -0.6907665874004316), ('temp_s', 0.5194285269030161), ('is_holiday', -0.36935936042954887), ('wind_s', -0.3318707866851824), ('hum_s', -0.25781018974338876), ('season_spring', -0.06231432490887823)]
Top beta (poly):   [('temp_s', 0.5108896079403374), ('season_winter', -0.46019879596325997), ('is_holiday', -0.3670929377353062), ('wind_s', -0.32415468333239483), ('hum_s', -0.2672746247027816), ('season_summer', 0.20594370572135975), ('temp2_s', -0.15175112888378192)]

WAIC:
linear: Computed from 6000 posterior samples and 500 observations log-likelihood matrix.

              Estimate       SE
deviance_waic   621.41    31.40
p_waic            9.05        -
poly:  Computed from 6000 posterior samples and 500 observations log-likelihood matrix.

              Estimate       SE
deviance_waic   567.93    31.74
p_waic            9.82        -

LOO:
linear: Computed from 6000 posterior samples and 500 observations log-likelihood matrix.

             Estimate       SE
deviance_loo   621.44    31.41
p_loo            9.06        -
------

Pareto k diagnostic values:
                         Count   Pct.
(-Inf, 0.70]   (good)      500  100.0%
   (0.70, 1]   (bad)         0    0.0%
   (1, Inf)   (very bad)    0    0.0%

poly:  Computed from 6000 posterior samples and 500 observations log-likelihood matrix.

             Estimate       SE
deviance_loo   567.96    31.75
p_loo            9.83        -
------

Pareto k diagnostic values:
                         Count   Pct.
(-Inf, 0.70]   (good)      500  100.0%
   (0.70, 1]   (bad)         0    0.0%
   (1, Inf)   (very bad)    0    0.0%


LOO:
         rank    elpd_loo     p_loo  ...        dse  warning     scale
poly       0  567.964419  9.834534  ...   0.000000    False  deviance
linear     1  621.441678  9.059095  ...  14.157582    False  deviance

[2 rows x 9 columns]
```
