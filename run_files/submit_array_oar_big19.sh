#!/bin/bash
#OAR -l {host='big19'}/core=4,walltime=30:00:00
#OAR --notify mail:lou.moulin-roussel@lip6.fr
#OAR -O logs/approx_%jobid%.out
#OAR -E logs/approx_%jobid%.err

# =====================================================================
# Approximation OWA (Chen et al., annexe C) sur des instances d'affectation
# Un job = une configuration (n, delta, eps), 30 instances résolues en
# parallèle sur les 4 cœurs réservés, avec un budget total de TIME_LIMIT s.
# =====================================================================
#
# Paramètres : n [delta] [eps]   (delta et eps valent 0.1 par défaut)
#
# Le dossier logs/ doit exister AVANT la soumission : mkdir -p logs
#
# 1. Mode direct (test local) :
#    ./submit_array_oar_big17.sh 10 0.1 0.1
#
# 2. Mode OAR, un seul job :
#    oarsub -S "./submit_array_oar_big17.sh 10 0.1 0.1"
#
# 3. Mode OAR array (une ligne "n delta eps" par job dans params.txt) :
#    oarsub -S ./submit_array_oar_big17.sh --array-param-file params.txt
#
# =====================================================================

# Détecter le mode array OAR
ARRAY_MODE=0
if [ -n "$OAR_ARRAY_ID" ] && [ -n "$OAR_ARRAY_INDEX" ]; then
    ARRAY_MODE=1
fi

usage() {
    cat >&2 << EOF
Usage : $0 n [delta] [eps]
  n      taille du problème (obligatoire)
  delta  paramètre de la grille POS (défaut : 0.1)
  eps    paramètre de la grille des t (défaut : 0.1)
EOF
    exit 1
}

# =====================================================================
# Configuration
# =====================================================================

N=$1
DELTA=${2:-0.1}
EPS=${3:-0.1}
ALPHA=${4:-0.0}
W_TYPE="linear_weights"
NB_INSTANCES=30
TIME_LIMIT=55000          # budget total (s, temps réel) pour les NB_INSTANCES instances

[ -z "$N" ] && usage

PROJECT_DIR=$HOME/projet_owa_min_sum
SCRIPT=$PROJECT_DIR/approx_owa_min_sum/ordered_opt.py
RESULTS_DIR=$PROJECT_DIR/results/assignment
OUTPUT=$RESULTS_DIR/approx_n${N}_delta${DELTA}_eps${EPS}_alpha${ALPHA}_${W_TYPE}.parquet

if [ $ARRAY_MODE -eq 1 ]; then
    echo "[submit_array] n=${N} delta=${DELTA} eps=${EPS} w_type=${W_TYPE} (job ${OAR_ARRAY_INDEX}) -> ${OUTPUT}"
else
    echo "[submit_single] n=${N} delta=${DELTA} eps=${EPS} w_type=${W_TYPE} -> ${OUTPUT}"
fi

if [ -n "$OAR_NODEFILE" ]; then
    echo "Cœurs alloués par OAR : $(wc -l < "$OAR_NODEFILE") sur $(sort -u "$OAR_NODEFILE" | tr '\n' ' ')"
fi

# =====================================================================
# Environnement
# =====================================================================

source ~/venvs/gurobi-135/bin/activate

# Gurobi : inutile pour ordered_opt.py, à garder seulement si instance_generator en a besoin
export GUROBI_HOME=$HOME/gurobi/gurobi1300/linux64
export PATH=$GUROBI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$GUROBI_HOME/lib:$LD_LIBRARY_PATH
export GRB_LICENSE_FILE=$HOME/gurobi_licenses/big19.lic

# Un seul thread de calcul par processus worker (le parallélisme vient des workers)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$RESULTS_DIR"

# =====================================================================
# Appel au script Python
# =====================================================================

python3 "$SCRIPT" \
    --n "$N" \
    --delta "$DELTA" \
    --eps "$EPS" \
    --alpha "$ALPHA" \
    --w-type "$W_TYPE" \
    --nb-instances "$NB_INSTANCES" \
    --time-limit "$TIME_LIMIT" \
    --output "$OUTPUT"
STATUS=$?

if [ $STATUS -eq 0 ]; then
    echo "✅ Job terminé : $OUTPUT"
else
    echo "❌ Échec du script Python (code $STATUS)" >&2
fi
exit $STATUS
