# Agente humanoide en MuJoCo con aprendizaje por refuerzo

Proyecto del curso **Aprendizaje por Refuerzo** (Universidad de Concepción, 2025), profesor Julio Godoy. Trabajo en grupo: Mirko Peñailillo, Javier Reyes y Mario Salgado.

Comparamos tres algoritmos de aprendizaje por refuerzo profundo, **PPO**, **SAC** y **TD3**, implementados en PyTorch, para controlar un humanoide de 17 articulaciones en el simulador **MuJoCo** (`Humanoid` de Gymnasium). El objetivo era que el agente aprendiera a **mantener el equilibrio**, **caminar** y **resistir perturbaciones** como viento, suelo resbaladizo o pendientes.

## Métodos

| Algoritmo | Implementación |
|---|---|
| **PPO** (Proximal Policy Optimization) | Actor-crítico con política Tanh-Gaussiana, *clipping* (ε = 0.2), GAE (γ = 0.995, λ = 0.95), entornos vectorizados en paralelo, *early stopping* por divergencia KL y learning rate decreciente |
| **SAC** (Soft Actor-Critic) | Doble crítico (*twin Q-networks*) para reducir la sobreestimación, política Tanh-Gaussiana y ajuste automático de la entropía |
| **TD3** (Twin Delayed DDPG) | Actor determinista, doble crítico, *target policy smoothing* y actualizaciones retrasadas. *Curriculum learning* (primero equilibrio, luego caminata) y *reward shaping* según el avance, el contacto de los pies con el suelo y la estabilidad del torso |

Para evaluar la robustez, PPO incluye un **ambiente modificado** con cuatro niveles de dificultad (`easy`, `medium`, `hard` y `extreme`) que combinan viento lateral, baja fricción y pendientes. SAC se evaluó aplicando una fuerza de viento sobre el torso.

## Resultados

- **SAC** aprende rápido: a los 400 mil pasos ya mantiene el equilibrio y a los 1,6 millones camina de forma coordinada. Pero es inestable: cerca de los 900 mil pasos su recompensa cayó de ~4.000 a ~2.000, y el viento logra derribarlo.
- **PPO** resultó el más robusto para la locomoción. El *early stopping* por KL evita actualizaciones demasiado bruscas y el learning rate decreciente permite un control fino al final del entrenamiento.
- **TD3** necesitó un entrenamiento progresivo (equilibrio → caminata) para no quedar atrapado en malas soluciones.

**Conclusión:** no hay un algoritmo óptimo para todo. En control continuo es clave entrenar de forma progresiva y **observar el comportamiento del agente**, no solo la recompensa, para detectar soluciones no deseadas.

### Videos

| Video | Contenido |
|---|---|
| [`HumanoidSinEntrenar.mp4`](Videos/HumanoidSinEntrenar.mp4) | El humanoide antes de entrenar |
| [`SAC_Balance.mp4`](Videos/SAC_Balance.mp4) | SAC manteniendo el equilibrio |
| [`SAC_Caminar.mp4`](Videos/SAC_Caminar.mp4) | SAC caminando |
| [`SAC_Robustez.mp4`](Videos/SAC_Robustez.mp4) | SAC bajo viento |
| [`PPO_Robustez.mp4`](Videos/PPO_Robustez.mp4) | PPO en el ambiente modificado |

La presentación completa, con las curvas de entrenamiento, está en [`AgenteHumanoideEnMujocoFinal_.pdf`](AgenteHumanoideEnMujocoFinal_.pdf).

## Estructura

```
├── PPO/ppo_funcional_vectorized.py   entrenamiento y evaluación de PPO (ambiente normal y modificado)
├── SAC/
│   ├── sac.py                        entrenamiento de SAC
│   ├── visualize_sac.py              visualización del último modelo
│   └── viento.py                     evaluación con viento sobre el torso
├── TD3/
│   ├── td3_train.py                  entrenamiento de TD3 (etapas balance y walk)
│   └── td3_watch.py                  visualización de un checkpoint
├── Videos/                           ejecuciones de los agentes
└── AgenteHumanoideEnMujocoFinal_.pdf presentación final
```

## Cómo correrlo

```bash
pip install "gymnasium[mujoco]" torch stable-baselines3 tensorboard tqdm
```

**PPO**
```bash
python ppo_funcional_vectorized.py --mode train                                  # entrenar desde cero
python ppo_funcional_vectorized.py --mode resume                                 # continuar desde un checkpoint
python ppo_funcional_vectorized.py --mode eval                                   # evaluar
python ppo_funcional_vectorized.py --mode eval_modified --difficulty medium      # evaluar en el ambiente modificado
```

**SAC**
```bash
python sac.py              # entrenar
python visualize_sac.py    # ver el último modelo entrenado
python viento.py           # evaluar con viento sobre el torso
```

**TD3**
```bash
python td3_train.py --run_name TD3_stage1_balance --stage balance    # etapa 1: equilibrio
python td3_train.py --run_name TD3_stage2_walk --stage walk          # etapa 2: caminata
python td3_train.py --run_name TD3_stage2_walk --stage walk \
  --resume runs_td3/TD3_stage2_walk/checkpoints/td3_step_5000000.pt --total_steps 20000000
python td3_watch.py --run_dir runs_td3/TD3_stage2_walk --target_steps 18000000 --episodes 3
```

## Referencias

- [ProfessorNova/PPO-Humanoid](https://github.com/ProfessorNova/PPO-Humanoid)
- [xbpeng/DeepMimic](https://github.com/xbpeng/DeepMimic)
- [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
