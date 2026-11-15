# Assets do SimuladorGarra

Organização das malhas e texturas usadas pelo backend MuJoCo e pelo frontend 3D.

## Estrutura

```text
assets/
├── menagerie/                 # Previews PNG do Google DeepMind MuJoCo Menagerie
│                              # (https://github.com/google-deepmind/mujoco_menagerie/tree/main/assets)
├── Garra/                     # Modelos 3D de braços e garras (malhas STL/OBJ + XML MJCF)
│   ├── robotiq_2f85/          # Garra paralela industrial Robotiq 2F-85
│   ├── robotiq_2f85_v4/       # Variante v4 + sensor de força
│   ├── universal_robots_ur5e/ # Braço 6 eixos UR5e (estilo do simulador online)
│   ├── low_cost_robot_arm/    # Braço educacional open-source com garra
│   └── umi_gripper/           # Garra UMI
├── moveis/                    # Móveis / bancada / célula (ainda vazio)
└── objetos/
    └── odontologia/           # Modelos 3D dentais (dentes, ferramentas, moldes)
```

## Origem e licença (Menagerie)

Os pacotes em `Garra/` e os PNGs em `menagerie/` vêm do
[google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(licença Apache-2.0 — ver `menagerie/LICENSE`). Cada subpacote pode ter
atribuições adicionais no próprio diretório (`LICENSE` / README).

> **Importante:** a pasta `assets/` do Menagerie no GitHub contém **apenas
> imagens PNG de preview**, não as malhas 3D. As malhas reais ficam em
> `<robot>/assets/*.stl|*.obj` — por isso copiamos os pacotes completos
> para `assets/Garra/`.

## Como usar no MuJoCo

Exemplo mínimo apontando para a Robotiq já baixada:

```xml
<!-- Em um MJCF seu -->
<compiler meshdir="../assets/Garra/robotiq_2f85/assets" texturedir="../assets/Garra/robotiq_2f85/assets"/>
<include file="../assets/Garra/robotiq_2f85/2f85.xml"/>
```

Ou carregue direto no Python:

```python
import mujoco
model = mujoco.MjModel.from_xml_path("assets/Garra/robotiq_2f85/scene.xml")
```

## Odontologia — o que baixar e para onde

O Menagerie **não tem** modelos dentais. Coloque arquivos em:

```text
assets/objetos/odontologia/
```

Formatos preferidos (em ordem):

1. **`.glb` / `.gltf`** — melhor para o frontend WebGL/Three.js  
2. **`.obj` + `.mtl`** — fácil de converter / visualizar  
3. **`.stl`** — ótimo para colisão MuJoCo; visual pobre (sem cor)

### Fontes gratuitas recomendadas (licença livre / CC)

| Fonte | O que buscar | Licença típica |
|-------|--------------|----------------|
| [NIH 3D](https://3d.nih.gov/) | “tooth”, “dental”, “molar”, “jaw” | Aberta (verificar cada asset) |
| [Sketchfab](https://sketchfab.com/search?q=dental+tooth&features=downloadable&licenses=322a749bcfa841b29dff1e8a1bb74b0b) | dente, molar, maxila, broca | Filtrar **Downloadable + CC** |
| [Poly Haven](https://polyhaven.com/) | props genéricos (não dentais) | CC0 |
| [Thingiverse](https://www.thingiverse.com/search?q=dental+tooth) | modelos STL dentais | Creative Commons (verificar) |

### Checklist ao baixar um modelo dental

1. Confirme a **licença** (CC0 / CC-BY / Apache — evite “não comercial” se for publicar).  
2. Exporte ou baixe em **GLB** (visual) e, se possível, um **STL** simplificado (colisão).  
3. Salve com nomes claros, por exemplo:  
   `molar_superior.glb`, `molar_superior_collision.stl`, `broca_alta_rotacao.glb`.  
4. Escala: meça no Blender e normalize para **metros** (MuJoCo e o frontend usam metros).  
5. Reduza polígonos (Blender → Decimate) para &lt; ~20k triângulos por peça no browser.

### Exemplo de pastas após baixar

```text
assets/objetos/odontologia/
├── dente_molar.glb
├── dente_molar_collision.stl
├── instrument_probe.glb
└── LICENSE_ATTRIBUTION.txt   # cite autores / links
```
