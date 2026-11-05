#!/bin/bash
#OAR -l {host='big17'}/core=4,walltime=30:00:00
#OAR --notify mail:lou.moulin-roussel@lip6.fr
#OAR -O logs/min_lorenz_%jobid%.out
#OAR -E logs/min_lorenz_%jobid%.err

# =====================================================================
# min_lorenz.py : exact (Gurobi), min_lorenz, min_lorenz_v2
# Un job = une configuration (méthode, n, eps), NB_INSTANCES instances résolues
# en parallèle sur les 4 cœurs réservés, avec un budget total de TIME_LIMIT s.
# =====================================================================
#
# Paramètres : method n [eps]
#   method  exact | min_lorenz | min_lorenz_v2
#   n       taille du problème
#   eps     arrondi géométrique des coûts pour min_lorenz (défaut 0 = aucun ; ignoré pour exact)
#
# Le dossier logs/ doit exister AVANT la soumission : mkdir -p logs
#
# 1. Mode direct (test local) :
#    ./submit_min_lorenz_oar_big17.sh min_lorenz 10 0
#
# 2. Mode OAR, un seul job :
#    oarsub -S "./submit_min_lorenz_oar_big17.sh exact 10"
#
# 3. Mode OAR array (une ligne "method n [eps]" par job dans params_min_lorenz.txt) :
#    oarsub -S ./submit_min_lorenz_oar_big17.sh --array-param-file params_min_lorenz.txt
#
#    Exemple de params_min_lorenz.txt :
#      exact 10
#      min_lorenz 10 0
#      min_lorenz 10 0.01
#      min_lorenz_v2 10 0
#
# =====================================================================

ARRAY_MODE=0
if [ -n "$OAR_ARRAY_ID" ] && [ -n "$OAR_ARRAY_INDEX" ]; then
    ARRAY_MODE=1
fi

usage() {
    cat >&2 << EOF
Usage : $0 method n [eps]
  method  exact | min_lorenz | min_lorenz_v2
  n       taille du problème
  eps     arrondi des coûts pour min_lorenz (défaut : 0)
EOF
    exit 1
}

# =====================================================================
# Configuration
# =====================================================================

METHOD=$1
N=$2
EPS=${3:-0}
W_TYPE="linear_weights"
NB_INSTANCES=30
TIME_LIMIT=55000          # budget total (s, temps réel) pour les NB_INSTANCES instances
GUROBI_THREADS=1         # par worker : les 4 cœurs sont déjà utilisés par 4 instances en parallèle
GUROBI_MEM_GB=8          # limite mémoire Gurobi PAR worker, en Go (vide = aucune limite)

[ -z "$METHOD" ] || [ -z "$N" ] && usage
case "$METHOD" in
    exact|min_lorenz|min_lorenz_v2) ;;
    *) echo "Méthode inconnue : $METHOD" >&2; usage ;;
esac

PROJECT_DIR=$HOME/projet_owa_min_sum
SCRIPT=$PROJECT_DIR/approx_owa_min_sum/min_lorenz.py
RESULTS_DIR=$PROJECT_DIR/results/assignment
INSTANCES_DIR=$PROJECT_DIR/results/instances

if [ $ARRAY_MODE -eq 1 ]; then
    echo "[submit_array] method=${METHOD} n=${N} eps=${EPS} w_type=${W_TYPE} (job ${OAR_ARRAY_INDEX})"
else
    echo "[submit_single] method=${METHOD} n=${N} eps=${EPS} w_type=${W_TYPE}"
fi

if [ -n "$OAR_NODEFILE" ]; then
    echo "Cœurs alloués par OAR : $(wc -l < "$OAR_NODEFILE") sur $(sort -u "$OAR_NODEFILE" | tr '\n' ' ')"
fi

# =====================================================================
# Environnement
# =====================================================================

source ~/venvs/gurobi-135/bin/activate

# Gurobi : nécessaire pour --method exact
export GUROBI_HOME=$HOME/gurobi/gurobi1300/linux64
export PATH=$GUROBI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$GUROBI_HOME/lib:$LD_LIBRARY_PATH
export GRB_LICENSE_FILE=$HOME/gurobi_licenses/big17.lic

# Un seul thread de calcul par processus worker
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$RESULTS_DIR" "$INSTANCES_DIR"

# =====================================================================
# Appel au script Python
# =====================================================================

python3 "$SCRIPT" \
    --method "$METHOD" \
    --n "$N" \
    --eps "$EPS" \
    --w-type "$W_TYPE" \
    --nb-instances "$NB_INSTANCES" \
    --time-limit "$TIME_LIMIT" \
    --gurobi-threads "$GUROBI_THREADS" \
    ${GUROBI_MEM_GB:+--gurobi-mem-gb "$GUROBI_MEM_GB"} \
    --results-dir "$RESULTS_DIR" \
    --instances-dir "$INSTANCES_DIR"
STATUS=$?

if [ $STATUS -eq 0 ]; then
    echo "✅ Job terminé : ${METHOD} n=${N} eps=${EPS}"
else
    echo "❌ Échec du script Python (code $STATUS)" >&2
fi
exit $STATUS
