"""Download PDB structures for the enzyme design benchmark set.

Downloads 2B3P (GFP from htFuncLib paper) plus a curated set of ~40
well-studied enzymes with known catalytic residues documented in M-CSA.

Usage:
    conda run -n quris python scripts/data_prep/download_pdbs.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import urllib.request
import urllib.error
import time

from src.utils.feature_cache import FeatureCache, CacheMetadata
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Curated enzyme benchmark set
# Each entry: (PDB_ID, enzyme_name, EC_number, fold_family, chain)
# Catalytic residues sourced from M-CSA (https://www.ebi.ac.uk/thornton-srv/m-csa/)
# ---------------------------------------------------------------------------
ENZYME_SET = [
    # === GFP (htFuncLib target) ===
    ("2B3P", "Green Fluorescent Protein", "N/A", "beta_barrel", "A"),

    # === TIM barrel enzymes ===
    ("1TIM", "Triosephosphate Isomerase", "5.3.1.1", "TIM_barrel", "A"),
    ("1ALD", "Fructose-bisphosphate Aldolase", "4.1.2.13", "TIM_barrel", "A"),
    ("1ENO", "Enolase", "4.2.1.11", "TIM_barrel", "A"),
    ("1PII", "Phosphotriesterase", "3.1.8.1", "TIM_barrel", "A"),
    ("1BTL", "Beta-lactamase TEM-1", "3.5.2.6", "alpha_beta", "A"),
    ("1XNB", "Xylanase", "3.2.1.8", "TIM_barrel", "A"),
    ("7TIM", "Triosephosphate Isomerase (yeast)", "5.3.1.1", "TIM_barrel", "A"),
    ("1TPH", "Tryptophan Synthase", "4.2.1.20", "TIM_barrel", "A"),
    ("2MNR", "Mandelate Racemase", "5.1.2.2", "TIM_barrel", "A"),

    # === Serine proteases ===
    ("4CHA", "Chymotrypsin", "3.4.21.1", "serine_protease", "A"),
    ("1S0Q", "Trypsin", "3.4.21.4", "serine_protease", "A"),
    ("1SBN", "Subtilisin Novo", "3.4.21.62", "subtilisin", "A"),
    ("1EAI", "Elastase", "3.4.21.36", "serine_protease", "E"),
    ("3TEC", "Thermitase", "3.4.21.66", "subtilisin", "A"),

    # === Cysteine proteases ===
    ("1POP", "Papain", "3.4.22.2", "papain_like", "A"),
    ("1AIM", "Caspase-1", "3.4.22.36", "caspase", "A"),

    # === Lipases / esterases ===
    ("1LPB", "Lipase B (Candida rugosa)", "3.1.1.3", "alpha_beta_hydrolase", "A"),
    ("1TCA", "Lipase (Candida antarctica B)", "3.1.1.3", "alpha_beta_hydrolase", "A"),
    ("1QJT", "Cutinase", "3.1.1.74", "alpha_beta_hydrolase", "A"),
    ("2LIP", "Lipase (Rhizomucor miehei)", "3.1.1.3", "alpha_beta_hydrolase", "A"),

    # === Glycosidases ===
    ("1BVV", "Beta-glucosidase", "3.2.1.21", "TIM_barrel", "A"),
    ("1CEL", "Cellulase Cel5A", "3.2.1.4", "TIM_barrel", "A"),
    ("1HEW", "Hen Egg-White Lysozyme", "3.2.1.17", "lysozyme", "A"),
    ("2LZM", "T4 Lysozyme", "3.2.1.17", "lysozyme", "A"),
    ("3LZM", "T4 Lysozyme L99A", "3.2.1.17", "lysozyme", "A"),
    ("1GOX", "Glucose Oxidase", "1.1.3.4", "TIM_barrel", "A"),

    # === Oxidoreductases ===
    ("1LDG", "Lactate Dehydrogenase", "1.1.1.27", "rossmann", "A"),
    ("3ADH", "Alcohol Dehydrogenase", "1.1.1.1", "rossmann", "A"),
    ("1GRB", "Glutathione Reductase", "1.8.1.7", "rossmann", "A"),

    # === Kinases and transferases ===
    ("2PKA", "Protein Kinase A", "2.7.11.11", "protein_kinase", "E"),
    ("1HCK", "CDK2", "2.7.11.22", "protein_kinase", "A"),

    # === Metalloenzymes ===
    ("1CA2", "Carbonic Anhydrase II", "4.2.1.1", "carbonic_anhydrase", "A"),
    ("1THL", "Thermolysin", "3.4.24.27", "metalloprotease", "A"),
    ("2CPP", "Cytochrome P450cam", "1.14.15.1", "cytochrome_p450", "A"),

    # === Designed enzymes ===
    ("2RKX", "Kemp Eliminase KE07", "4.2.1.-", "TIM_barrel", "A"),
    ("3IIO", "Kemp Eliminase KE70", "4.2.1.-", "TIM_barrel", "A"),
    ("4A29", "Retro-aldolase RA95", "4.1.2.-", "TIM_barrel", "A"),

    # === Classic textbook enzymes ===
    ("1LZA", "Lysozyme (human)", "3.2.1.17", "lysozyme", "A"),
    ("3RN3", "Ribonuclease A", "3.1.27.5", "rnase", "A"),
    ("1ACB", "Alpha-chymotrypsin (complex)", "3.4.21.1", "serine_protease", "E"),
]


def download_pdb(pdb_id: str, output_dir: Path) -> Path:
    """Download a single PDB file from RCSB.

    Args:
        pdb_id: 4-character PDB identifier
        output_dir: Directory to save the file

    Returns:
        Path to the downloaded file

    Raises:
        Exception on download failure
    """
    pdb_id_upper = pdb_id.upper()
    url = f"https://files.rcsb.org/download/{pdb_id_upper}.pdb"
    output_path = output_dir / f"{pdb_id_upper}.pdb"

    if output_path.exists():
        logger.info(f"Already downloaded: {pdb_id_upper}")
        return output_path

    logger.info(f"Downloading {pdb_id_upper} from {url}")
    urllib.request.urlretrieve(url, str(output_path))
    return output_path


def main():
    pdb_dir = project_root / "data" / "pdb"
    pdb_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = project_root / "cache" / "pdb_downloads"
    cache = FeatureCache(str(cache_dir))

    logger.info(f"Downloading {len(ENZYME_SET)} PDB structures to {pdb_dir}")

    downloaded = []
    failed = []

    for pdb_id, name, ec, fold, chain in ENZYME_SET:
        cache_key = {"pdb_id": pdb_id, "method": "pdb_download"}

        try:
            pdb_path = download_pdb(pdb_id, pdb_dir)

            # Cache metadata about this download
            if not cache.has(cache_key):
                cache.save(
                    cache_key,
                    {
                        "pdb_id": pdb_id,
                        "name": name,
                        "ec_number": ec,
                        "fold_family": fold,
                        "chain": chain,
                        "path": str(pdb_path),
                    },
                    metadata=CacheMetadata(
                        method="rcsb_download",
                        source=pdb_id,
                        params={"ec": ec, "fold": fold},
                    ),
                )

            downloaded.append(pdb_id)
            logger.info(f"  OK: {pdb_id} - {name}")

        except Exception as e:
            failed.append((pdb_id, str(e)))
            logger.warning(f"  FAILED: {pdb_id} - {name}: {e}")

        # Be polite to RCSB servers
        time.sleep(0.25)

    logger.info(
        f"\nDone. Downloaded: {len(downloaded)}, Failed: {len(failed)}"
    )
    if failed:
        logger.warning("Failed downloads:")
        for pdb_id, err in failed:
            logger.warning(f"  {pdb_id}: {err}")


if __name__ == "__main__":
    main()
