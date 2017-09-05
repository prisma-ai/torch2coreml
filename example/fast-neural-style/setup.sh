if [ ! -d "fast-neural-style" ]; then
  echo "Cloning fast-neural-style repo"
  git clone https://github.com/jcjohnson/fast-neural-style.git
fi

cd fast-neural-style

if [ -z "$(ls models/eccv16)" ]; then
  echo "Downloading pretrained style-transfer models"
  bash models/download_style_transfer_models.sh
fi

cd ..

# Prepare models
echo "Preparing models for conversion"
mkdir -p prepared_models
for f in fast-neural-style/models/instance_norm/*.t7
do
  th prepare_model.lua -input $f -output prepared_models/$(basename $f)
done

# Convert style-transfer models to CoreML
echo "Converting models to CoreML"
mkdir -p coreml_models
for f in prepared_models/*.t7
do
  echo "Converting $f"
  python convert-fast-neural-style.py -input $f -output coreml_models/$(basename $f .t7).mlmodel
done
