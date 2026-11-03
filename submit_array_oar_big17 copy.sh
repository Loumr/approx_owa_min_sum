#!/bin/bash
#OAR -l {host='big17'}/core=4,walltime=120:00:00
#OAR --notify mail:lou.moulin-roussel@lip6.fr

# =====================================================================
# NOUVEAU SCRIPT OAR
# =====================================================================
#
#   ✅ Résultat : les trois variantes (no_filter, k_sum_no_del, k_sum_del)
#      sont AUTOMATIQUEMENT lancées dans un seul job pour chaque n
#
#   ✅ Sortie : UN SEUL fichier parquet par (n, delta, eps, w_type) 
#
# =====================================================================
# MODES DE FONCTIONNEMENT :
# =====================================================================
#
# 1. Mode DIRECT (test local) :
#    ───────────────────────────
#    ./submit_array_oar_elicitation_big17.sh 30
#
# 2. Mode OAR simple job (un seul job) :
#    ────────────────────────────────────
#    oarsub -l {host='big17'}/core=4,walltime=120:00:00 -S ./submit_array_oar_elicitation_big17.sh -- 30
#
# 3. Mode OAR array (multiple jobs via params.txt) :
#    ───────────────────────────────────────────────
#    oarsub -S ./submit_array_oar_elicitation_big17.sh --array-param-file params.txt
#
# =====================================================================

# Détecter le mode de fonctionnement OAR array
ARRAY_MODE=0
if [ -n "$OAR_ARRAY_ID" ] && [ -n "$OAR_ARRAY_INDEX" ]; then
    ARRAY_MODE=1
fi

# Valider qu'on a des paramètres
if [ $# -eq 0 ] && [ $ARRAY_MODE -eq 0 ]; then
    cat >&2 << 'EOF'
Erreur : aucun paramètre fourni et pas en mode array OAR.

Usage (mode direct) :
  ./submit_array_oar_elicitation_NOUVEAU.sh n
  
Usage (mode OAR simple job) :
  oarsub -l {host='big17'}/core=4,walltime=120:00:00 \
         -S ./submit_array_oar_elicitation_NOUVEAU.sh -- 30

Usage (mode OAR array) :
  oarsub -S ./submit_array_oar_elicitation_NOUVEAU.sh --array-param-file params.txt

Paramètres :
  n = taille du problème (obligatoire)

Note : Les trois variantes (no_filter, k_sum_no_del, k_sum_del) sont
lancées AUTOMATIQUEMENT et les résultats sont stockés dans UN SEUL fichier.

EOF
    exit 1
fi

# =====================================================================
# Configuration simplifié (ancien : 5 paramètres, maintenant : 1 seul)
# =====================================================================

N=$1
DELTA=$2
EPS=$3
W_TYPE="linear_weights"

if [ -z "$N" ]; then
    echo "Usage : $0 n" >&2
    exit 1
fi

# Nom du fichier de sortie 
OUTPUT=../results/assignment/approx_n${N}_delta${DELTA}_eps${EPS}_${W_TYPE}.parquet

if [ $ARRAY_MODE -eq 1 ]; then
    echo "[submit_array] n=${N} w_type=${W_TYPE} (job ${OAR_ARRAY_INDEX}/${OAR_ARRAY_NB}) -> ${OUTPUT}"
else
    echo "[submit_single] n=${N} w_type=${W_TYPE} -> ${OUTPUT}"
fi

# =====================================================================
# Configuration environnement
# =====================================================================

# Activate your Python virtual environment
source ~/venvs/gurobi-135/bin/activate

# Set gurobi variables
export GUROBI_HOME=$HOME/gurobi/gurobi1300/linux64
export PATH=$GUROBI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$GUROBI_HOME/lib:$LD_LIBRARY_PATH
export GRB_LICENSE_FILE=$HOME/gurobi_licenses/big17.lic

mkdir -p ../results/assignment

# =====================================================================
# APPEL AU SCRIPT PYTHON
# =====================================================================

python3 ~/projet_owa_min_sum/approx_owa_min_sum/ordered_opt.py \
    --n-values $N \
    --delta $DELTA \
    --eps $EPS \
    --w-type $W_TYPE \
    --nb-instances 30 \
    --time-limit 1900 \
    --output "$OUTPUT"

echo "✅ Job terminé : $OUTPUT"