**Attention** and sorry that in the coding process, I confused "PCG" and "mRNA" in the naming, so all "mRNA"s in the code means "PCG"s in the paper!

# My Final Solution on the Dataset Construction

Greatly thanks to Chengzhou Ouyang, I decide to build a miRNA-mRNA-disease heterogeneous graph.  

|Node|DataBase|Node Info|Comment|
|---|---|---|---|
|miRNA|miRBase|Sequence, Family|Authoritative|
|mRNA|HGNC|Gene Symbol, Group|Authoritative|
|disease|MeSH|Semantic Text, Father-son|Authoritative|

|Association|DataBase|Edge Info|RNA Symbol / Disease Name Matched|Comment|
|---|---|---|---|---|
|miRNA-disease|RNADisease|PMID*|miRBase / MeSH|Large scale, High quality|
|miRNA-mRNA|ENCORI Degradome-RNA|Experimental Information|miRBase / Unmatched|Authoritative|
|mRNA-disease|DisGeNet|Score|OMIM / MeSH / Entrez|Large scale, High quality|

\* : We can get the information of PubMed publications using the package "BioPython" (https://biopython.org/).  

# Details of Source Databases

## DisGeNET  

The following data files are downloaded from https://www.disgenet.org/downloads, please sign up and login first. And there is an about page: https://www.disgenet.org/dbinfo, read it for more information.  

- **disease_mapping.tsv**: Mappings from UMLS concept unique identifier to disease vocabularies: DO, EFO, HPO, ICD9CM, MSH, NCI, OMIM, and ORDO.  
- **disgenet_2020.db**: An important database file. Since there is a login restriction, please **download from DisGeNET by yourself**.


## ENCORI

The website of ENCORI is https://rna.sysu.edu.cn/encori/, and the data file is downloaded through API: https://rna.sysu.edu.cn/encori/tutorialAPI.php.  

There are 8 types of API, and we mainly focus on "Degradome-RNA".  
- **Degradome-RNA**: Get Data for **MiRNAs Cleavage Events**. Retrieve data for cleavage events of miRNAs on genes supported by degradome-seq data.
    - Query Format
        - assembly=[genome version]. e.g., hg19
        - geneType=[main gene type]: mRNA, ncRNA
        - miRNA=[microRNA name]. e.g., hsa-miR-196a-5p ("all" for downloading all regulatory data)
        - degraExpNum=[integer]: minimum number of supporting degradome-seq experiments
        - target=[gene name]. e.g., TP53 ("all" for downloading all regulatory data)
        - cellType=[cell type]. e.g., HeLa ("all" for downloading all regulatory data)
    - **mRNA_ENCORI_hg19_degradome-seq_all_all.txt**, geneType=mRNA: https://rna.sysu.edu.cn/encori/api/degradomeRNA/?assembly=hg19&geneType=mRNA&miRNA=all&degraExpNum=1&target=all&cellType=all  


## RNADisease

The data file is downloaded from http://www.rnadisease.org/download, read http://www.rnadisease.org/help for more details.  

- **RNADiseasev4.0_RNA-disease_experiment_miRNA.xlsx**


## MeSH

The MeSH data is downloaded from: https://www.nlm.nih.gov/databases/download/mesh.html, and we only adopt the Descriptor.   

- c2022.bin: ASCII MeSH. Supplementary Concept Records   
- d2022.bin: ASCII MeSH. **Descriptor**  
- q2022.bin: ASCII MeSH. Qualifier  

According to https://www.nlm.nih.gov/mesh/dtype.html, the data elements of Descriptor records in ASCII MeSH are noted below, then the highlighted elements are utilized for contructing our heterogeneous graph.    

|Abbr.   |   Full Title|
|----    |   ----|
|AN	|   Annotation|
|AQ  |	Allowable topical Qualifiers|
|CATSH   |	CataLoging SubHeadings list name|
|CX  |	Consider also Xref|
|DA  |	DAte of entry|
|DC  |	Descriptor Class|
|DE  |	Descriptor Entry version|
|DS |	Descriptor Sort version|
|DX |	Date major descriptor EStablished|
|EC |	Entry Combination|
|<mark>**PRINT ENTRY**</mark>    |	<mark>Entry Term, Print</mark>|
|<mark>**ENTRY**</mark>  |	<mark>Entry Term, Non-Print</mark>|
|<mark>**FX**</mark> |	<mark>Forward Cross reference (see also reference)</mark>|
|HN |	History Note|
|<mark>**MH**</mark> |	<mark>MeSH Heading</mark>|
|MH_TH  |	MeSH Heading Thesaurus ID [= MHTH in ELHILL MeSH]|
|<mark>**MN**</mark> |	<mark>MeSH tree Number</mark>|
|MR |	Major Revision date|
|<mark>**MS**</mark> |	<mark>MeSH Scope note</mark>|
|N1 |	CAS Type 1 name|
|OL |	OnLine note|
|PA |	Pharmacological Action|
|PI |	Previous Indexing|
|PM |	Public MeSH note|
|RECTYPE    |	RECord TYPE [= RY in ELHILL MeSH ]|
|RH |	Running Head, MeSH tree structures|
|RN |	CAS registry / EC number / UNII code|
|RR |	Related Registry number|
|ST |	Semantic Type|
|<mark>**UI**</mark> |	<mark>Unique Identifier</mark>|

According to https://meshb.nlm.nih.gov/treeView, element **MN (MeSH tree Number)** presents the category of a term and the tree structure between terms:  
- The first letter indicates category of the term. Importantly, we focus on the **Diseases [C]**.
- The numbers separated by dots indicate dependencies between terms. For diseases, it means the superclass-subclass (father-son) relationship.


## miRBase

The miRBase data is downloaded from miRBase v22.1: https://mirbase.org/ftp/CURRENT/.

- **aliases.txt**: aliases of miRNA symbols
- **hairpin.fa**: Fasta format sequences of all miRNA hairpins
- **mature.fa**: Fasta format sequences of all mature miRNA sequences
- **miRNA.xls**: "Pre-miRNA - Mature miRNA 1 - Mature miRNA 2" matchings, including the Accession, ID and Sequence fields

Specially, the miRNA family data is downloaded from miRBase v21: https://www.mirbase.org/ftp/21/.

- **miFam.dat**: miRNA families  


## HGNC

The  **protein-coding_gene.txt** are downloaded from database HGNC: https://www.genenames.org/download/statistics-and-files/.  

We mainly focus on these data fields:
- hgnc_id
- symbol
- name
- locus_type
- alias_symbol
- alias_name
- prev_symbol
- prev_name
- entrez_id


## HumanNet

The data file is downloaded from: https://www.inetbio.org/humannet/download.php. For more details, see the tutorial page: https://www.inetbio.org/humannet/tutorial.php.  

- **HumanNet-FN.tsv**: Functional gene network, Gene ID refers to Entrez.  