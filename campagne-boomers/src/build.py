import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import poster, avatar
import scenes

OUT = pathlib.Path(__file__).resolve().parent.parent / "html"
OUT.mkdir(exist_ok=True)

POSTERS = {
    "01-fax": dict(
        bg="linear-gradient(170deg,#EEDDBA 0%,#E2C796 100%)",
        question="Votre fax sonne dans le vide depuis 2009&#8239;?",
        scene_svg=scenes.fax(), worker="Hugo",
        avatar_svg=avatar("#f1c6a4", "#5b3a24", "#d9c6a3", "short"),
        messages=[("in", "Salut ! Je suis le Worker Hugo !<br>Ton devis ? Envoyé, signé et relancé."),
                  ("in", "Le fax, je connais pas 😅"), ("out_voice", "0:32"), ("in_voice", "1:05", "🤩")]),
    "02-mails-imprimes": dict(
        bg="linear-gradient(170deg,#F7CDBF 0%,#EFAA98 100%)",
        question="Vous imprimez vos mails pour les lire&#8239;?",
        scene_svg=scenes.mails(), worker="Inès",
        avatar_svg=avatar("#e8b48e", "#2b1a12", "#f3c9bd", "bun"),
        messages=[("in", "Salut ! Je suis le Worker Inès !<br>Je lis tes mails, je trie et je te fais un résumé vocal."),
                  ("in", "Les arbres te disent merci 🌳"), ("out_voice", "0:32"), ("in_voice", "1:05", "😍")]),
    "03-excel": dict(
        bg="linear-gradient(170deg,#C3E7CF 0%,#9DD2B1 100%)",
        question="Votre CRM s’appelle FINAL_v12.xls&#8239;?",
        scene_svg=scenes.excel(), worker="Nina",
        avatar_svg=avatar("#c98d66", "#3a2216", "#bfe3cb", "curly"),
        messages=[("in", "Salut ! Je suis le Worker Nina !<br>Je range tes contacts et je relance tes prospects."),
                  ("in", "Et je fusionne jamais les cellules 🤞"), ("out_voice", "0:32"), ("in_voice", "1:05", "😎")]),
    "04-repondeur": dict(
        bg="linear-gradient(170deg,#F6DE97 0%,#EEC75E 100%)",
        question="Votre répondeur est plein depuis 1998&#8239;?",
        scene_svg=scenes.repondeur(), scene_transform="translate(0 170) scale(.84)", worker="Léo",
        avatar_svg=avatar("#f3cdb0", "#c9853f", "#f4dc9a", "short"),
        messages=[("in", "Salut ! Je suis le Worker Léo !<br>Envoie-moi un vocal, je m’occupe du reste."),
                  ("out_voice", "0:04", "« Allô ? ALLÔ ? Ça marche ce truc ? »"),
                  ("in", "Oui Jean-Claude, ça marche 😄<br>Je t’écoute !")]),
    "05-minitel": dict(
        bg="linear-gradient(170deg,#D6CCF1 0%,#B7A6E5 100%)",
        question="Vous prospectez encore sur le 3615&#8239;?",
        scene_svg=scenes.minitel(), scene_transform="translate(-40 140) scale(.85)", worker="Malik",
        avatar_svg=avatar("#8d5a3b", "#1c120c", "#d3c8f0", "short"),
        messages=[("in", "Salut ! Je suis le Worker Malik !<br>Je trouve des clients pertinents et je prends tes rendez-vous."),
                  ("in", "Sans Minitel, promis 😉"), ("out_voice", "0:32"), ("in_voice", "1:05", "🤩")]),
}
for name, cfg in POSTERS.items():
    (OUT / f"{name}.html").write_text(poster(**cfg))
print("ok")
