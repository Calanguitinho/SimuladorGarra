# Modelos 3D de Odontologia

Assets usados pela cena **DentalGrip Studio** (Three.js) e pelo instrumento manipulável no MuJoCo.

## Arquivos atuais

| Arquivo | Uso |
|---------|-----|
| `dental_chair_ostem.glb` | Cadeira odontológica (ambiente visual) |
| `metal_tray.glb` | Bandeja metálica (prop visual) |
| `dentist_tools.glb` | Kit de instrumentos (prop visual) |
| `dental_tool-nih3d.stl` | Instrumento manipulável (física MuJoCo + sync visual) |
| `Angled_Dental_Floss_Holder.stl` | Suporte de fio dental (física MuJoCo + sync visual) |

## Como o projeto usa

- Frontend carrega via `/assets/objetos/odontologia/...` (manifesto em `GET /api/scene-manifest`).
- MuJoCo referencia o STL em `backend/models/gripper.mjcf` (`meshdir` + escala `0.001` mm→m).
- Props GLB são só visuais; o STL sincroniza pose com o corpo `dental_tool` do backend.

## Licenças

Anote autor/licença em `LICENSE_ATTRIBUTION.txt` quando adicionar novos modelos (NIH 3D, Sketchfab CC0/CC-BY, etc.).
