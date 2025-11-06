# 🚀 Guía para Subir el Proyecto a GitHub

## Paso 1: Preparar el Repositorio Local

```bash
# Navegar al directorio del proyecto
cd /home/huaritex/Desktop/Deep_Learning_Cybersecurity

# Inicializar Git (si no está inicializado)
git init

# Agregar todos los archivos
git add .

# Hacer el primer commit
git commit -m "Initial commit: Deep Learning Cybersecurity Threat Detection"
```

## Paso 2: Crear Repositorio en GitHub

1. Ve a [GitHub](https://github.com)
2. Haz clic en el botón **"+"** en la esquina superior derecha
3. Selecciona **"New repository"**
4. Completa la información:
   - **Repository name**: `deep-learning-cybersecurity` (o el nombre que prefieras)
   - **Description**: "Neural network for cyber threat detection using PyTorch"
   - **Visibility**: Public o Private (según tu preferencia)
   - **NO** marques "Initialize this repository with a README" (ya lo tenemos)
5. Haz clic en **"Create repository"**

## Paso 3: Conectar y Subir

```bash
# Agregar el repositorio remoto (reemplaza 'yourusername' con tu usuario de GitHub)
git remote add origin https://github.com/yourusername/deep-learning-cybersecurity.git

# Verificar que se agregó correctamente
git remote -v

# Subir el código a GitHub
git push -u origin main

# Si tu rama principal se llama 'master' en lugar de 'main':
# git branch -M main
# git push -u origin main
```

## Paso 4: Verificar

1. Ve a tu repositorio en GitHub
2. Verifica que todos los archivos estén presentes:
   - ✅ README.md
   - ✅ hola.ipynb
   - ✅ requirements.txt
   - ✅ .gitignore
   - ✅ LICENSE
   - ✅ example_usage.py

## 📝 Comandos Útiles para el Futuro

### Agregar cambios nuevos
```bash
git add .
git commit -m "Descripción de tus cambios"
git push
```

### Ver el estado de tu repositorio
```bash
git status
```

### Ver el historial de commits
```bash
git log --oneline
```

### Crear una nueva rama
```bash
git checkout -b nueva-funcionalidad
```

### Volver a la rama principal
```bash
git checkout main
```

## 🔐 Configurar Autenticación (si es necesario)

Si GitHub te pide autenticación, tienes dos opciones:

### Opción 1: Personal Access Token
1. Ve a GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Genera un nuevo token con permisos `repo`
3. Usa el token como contraseña cuando Git te lo pida

### Opción 2: SSH (Recomendado)
```bash
# Generar clave SSH
ssh-keygen -t ed25519 -C "tu_email@example.com"

# Copiar la clave pública
cat ~/.ssh/id_ed25519.pub

# Agregar la clave en GitHub: Settings → SSH and GPG keys → New SSH key
# Cambiar la URL remota a SSH
git remote set-url origin git@github.com:yourusername/deep-learning-cybersecurity.git
```

## ✨ Personalizar el README

No olvides personalizar en `README.md`:
- [ ] Tu nombre de usuario de GitHub en los enlaces
- [ ] Tu nombre y correo electrónico en la sección de Contacto
- [ ] Tu nombre en la sección de Copyright
- [ ] Enlaces de redes sociales

## 📦 Archivos a Revisar Antes de Subir

- [x] README.md - Documentación completa
- [x] requirements.txt - Dependencias del proyecto
- [x] .gitignore - Archivos a ignorar
- [x] LICENSE - Licencia del proyecto
- [x] hola.ipynb - Notebook principal
- [x] example_usage.py - Script de ejemplo

## 🎯 Opcional: Agregar Badges al README

Puedes agregar badges personalizados al README:
- GitHub stars: `[![Stars](https://img.shields.io/github/stars/yourusername/deep-learning-cybersecurity.svg)](https://github.com/yourusername/deep-learning-cybersecurity/stargazers)`
- Issues: `[![Issues](https://img.shields.io/github/issues/yourusername/deep-learning-cybersecurity.svg)](https://github.com/yourusername/deep-learning-cybersecurity/issues)`

---

¡Listo! Tu proyecto estará disponible en GitHub para compartir con la comunidad 🎉
