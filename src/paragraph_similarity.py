from rapidfuzz import fuzz

def token_fuzzy_find(doc: str, para: str, *, window=4000, step=200, min_score=70):
    """
    Slides a relatively large window and uses token_set_ratio
    Returns (start, end, score) or None.
    """
    n = len(doc)
    best = (-1, None)
    for start in range(0, max(1, n - window + 1), step):
        w = doc[start:start+window]
        score = fuzz.partial_ratio(w, para)
        if score > best[0]:
            best = (score, start)
    score, start = best
    if start is None or score < min_score:
        return None

    return (start, min(n, start + window), score)

if __name__ == "__main__":
    test_text = """Webbin laajakaistaiset havainnot infrapuna-alueella mahdollistavat ylikulkevien planeettojen kaasukehien tarkkailun ennennäkemättömällä tarkkuudella. Mittaamalla planeetan kaasukehän läpi suodattuvaa valoa eri aallonpituuksilla, Webb voi määrittää hyvinkin tarkasti ylikulkevien planeettojen kaasukehän koostumuksia. TRAPPIST-1 -järjestelm
    ä tarjoaa yhden lähikohteen, jolle havainnot ovat erityisen hedelmällisiä. Voimme käyttää tähteä kiertävän planeettaseitsikon ylikulkuja hyväksemme mitataksemme ensimmäistä kertaa monin tavoin maankaltaisten ja -ko
    koisten, elämän vyöhykkeen planeettojen ominaisuuksia. Voimme selvittää niiden kaasukehien paksuudet ja havaita merkkejä hiilidioksidista, sekä veden olemassaolosta kielivästä vesihöyrystä planeettojen kaasukehissä
    . Saatamme saada selville merkkejä metaanista, kertoen mahdollisesta geologisesta aktiivisuudesta tai jopa ensimmäisiä havaintoja biokemiallisten prosessien aikaan saamasta kemiallisesta epätasapainotilasta ja site
    n, välillisesti, elämästä.\nKykenemme jo nykyisellään havaitsemaan erilaisia molekyylejä eksoplaneettojen kaasukehissä."""
    p_1_minor_deletions= """Wein laajakaistaiset havainnot infrapuna-alueella mahdollistavat ylikulkevien planeettojen kaasuhien tarkkailun ennennäkemäömällä tarkkuudella. Mittaamalla planeetan kaasukehän läpi suodattuvaa valoa eri aallonpituuksilla, Webb voi mää´rittää hyvinkin tarkasti ylikulkevien planeettojen kaasukehä koostumuksia. TRAPPIST-1 -järjestelm
    ä tarjoaa yhden lähikohteen, jolle havainnot ovat erityisen hedelmällisiä."""
    p_2_major_deletions = "Saatamme  merkkejä, kertoen tai jopa ensimmäisiä havaintoja prosessien aan saamasta kemiallisesta ja sitn, välillisesti, elämästä. jo nykyisellään havaitsemaan erilaisia molekyylejä eksoplaneettojen kaasukehissä."
    p_3_perfect = "Kykenemme jo nykyisellään havaitsemaan erilaisia molekyylejä eksoplaneettojen kaasukehissä."
    p_4_different_doc = """Ruotsin innovaatiopolitiikka; Suomestako mallia?\nRuotsi pyrkii edistämään innovaatioilmastoa sekä luomaan innovaatioille suotuisat olosuhteet. Valtiovalta toimii yhteistyössä elinkeinoelämän, ammatti
yhdistyksien, korkeakoulujen ja viranomaisten kanssa luodakseen pitkän tähtäyksen strategian Ruotsin innovatiivisuuden ja kilpailukyvyn kehittämiseksi. Ruotsin innovaatiostrategialla on neljä pääteemaa ja niiden al
la yhteensä kymmenen osa-aluetta. Innovaatiopolitiikka on kiinteässä yhteydessä EU:n yhteiseen kasvustrategiaan eli Lissabonin strategiaan.\nRuotsalainen suurpanostus innovatiivisiin tuotteisiin on PIE-p -ohjelma, 
Product Innovation Engineering Program, jonka tarkoituksena on parantaa ruotsalaista tuotekehitystä ja liike-elämän kilpailukykyä. On päätetty luoda myös toimintapaketti IT- ja telekommunikaatiotekniikan kansainväl
isen kilpailukyvyn edelleen kehittämiseksi. Myös terästeollisuuden ja metallurgian, puu- ja metsäteollisuuden, lento- ja avaruusteollisuuden sekä lääkkeiden, biotekniikan ja lääketieteen tekniikan tutkimukseen pano
stetaan. Ajoneuvoteollisuuskin halutaan saada maailman kärkeen. Ruotsissa on kuitenkin keskusteltu paljon maan innovaatiopolitiikan tehottomuudesta. Mediassa onkin toivottu Ruotsin ottavan oppia Suomen innovaatiopo
litiikasta.\nRuotsi saa uuden porvarihallituksen 6.10.2006. Uuden hallituksen innovaatioasioista vastaavan ministerin nimeä ei ole tällä hetkellä vielä tiedossa. Suuria muutoksia innovaatiopolitiikan peruslinjauk-s
iin ei ole luvassa. Panostukset pienyrityksiin saattavat kuitenkin heijastua myös innovaatiopolitiikkaan. Ruotsi pyrkii edistämään innovaatioilmastoa sekä luomaan innovaatioille suotuisat olosuhteet. Edellisen sosi
alidemokraattisen vähemmistöhallituksen yhteistyössä elinkeinoelämän, ammattiyhdistyksien, yliopistojen, korkeakoulujen ja viranomaisten kanssa kehittämät innovaatiojärjestelmät muodostuvat kaikista tutkimuksen, li
ike-elämän, politiikan ja julkisen toiminnan aloista, jotka toimivat uuden teknologian ja uuden tiedon tuottamiseksi, levittämiseksi ja soveltamiseksi. Innovaatiojärjestelmien määritelmään kuuluu myös se, että inno
vaatioilla pyritään luomaan kestävää kasvua uusien tuotteiden, palvelujen ja prosessien avulla.\nRuotsi pyrkii luomaan pitkän tähtäyksen strategian maan innovatiivisuuden ja kilpailukyvyn kehittämiseksi. Ruotsin in
novaatiostrategialla on neljä pääteemaa ja niiden alla yhteensä kymmenen osa-aluetta. Ensimmäinen pääteema eli innovaatioiden tietoperusta, jakautuu kolmeen osatekijään eli ruotsalaisen tutkimuksen ja koulutuksen p
itämiseen maailmanluokkaisena ja ruotsalaisiin profiilialoihin panostamiseen sekä globalisaation mahdollisuuksien hyödyntämiseen. Toinen pääteema on innovatiivinen elinkeinoelämä, joka tarkoittaa innovaatiostrategi
an mukaan erityisesti nokkelien pienten ja keskisuurten yritysten innovaatiokyvyn vahvistamista sekä tutkimustulosten ja ideoiden kaupallistamista. Kolmas pääteema on innovatiiviset julkiset investoinnit, jonka osa
-alueita ovat julkisen sektorin käyttäminen kestävän kehityksen moottorina, julkisen toiminnan uudistamisen ja tehokkuuden edistäminen sekä sellaisen infrastruktuurin kehittäminen, joka edistää uudistuksia ja kestä
vää kasvua. Neljäs ja viimeinen innovaatiostrategian pääteema ovat innovatiiviset ihmiset, jonka alateemoina on yrittäjyyden ja yritystoiminnan stimuloiminen ja se, että ihmisten osaamisen annetaan tulla esille ja 
oikeuksiinsa.\nLissabonin strategia\nInnovaatiopolitiikka on kiinteässä yhteydessä Euroopan unionin yhteiseen kasvustrategiaan, niin kutsuttuun Lissabonin strategiaan. EU:n tavoitteena on olla vuoteen 2010 mennessä
 maailman kilpailukykyisin tietotalous: tämä on tavoite, jonka tiedetään olevan varsin vaikea, ellei jopa mahdoton saavuttaa. EU pyrkii kestävään kehitykseen, sosiaaliseen yhteishenkeen ja saamaan aikaan sekä lisää
 että entistä parempia työmahdollisuuksia. Kysymys on kaikkien unionin kansalaisten hyvinvoinnin turvaamisesta. Kehityksen vauhdittamiseksi EU:n valtioiden ja jäsenmaiden hallituksien päämiehet päättivät, että joka
inen jäsenvaltio luo oman toimintaohjelman. EU-maat ovat päättäneet, että talous, hyvinvointi ja ympäristökysymykset käyvät käsi kädessä, eikä taloutta enää aseteta muiden kysymysten edelle. EU:n päätös on näin oll
en linjassa Ruotsin tavoitteiden kanssa. Ruotsissa on edellisen hallituksen taholta pelätty, että Lissabon-strategia keskittyisi liikaa talouskasvuun hyvinvointi- ja ympäristökysymysten kustannuksella.\nRuotsin toi
mintaohjelma kasvun ja työllisyyden edistämiseksi\nLokakuussa 2005 Ruotsin hallitus julkaisi toimintaohjelmansa kasvun ja työllisyyden edistämiseksi: Sveriges handlingsprogram för tillväxt och sysselsättning. Toimi
ntaohjelman toivotaan vastaavan globalisaation tuomiin haasteisiin. Globalisaation tuomat muutokset vaativat, että valtiot yhteisesti koko ajan uudelleenkokeilevat, uudistavat ja etsivät parempia ratkaisuja. Uusia 
haasteita ilmaantuu jatkuvasti, toimintaohjelmassa todetaan. Juuri globalisaatiolla perustellaan, että Ruotsikaan ei voi herpaantua, vaikka Ruotsi on useilla aloilla monia muita maita paljon edellä, kertoo Ruotsin 
toimintaohjelma. Ruotsin radion uutinen 20.3.2006 kertoo, kuinka EU:n johtajat keskustellessaan talouskasvureformista luokittelivat kaikki EU-maat \"sankari tai konna\" -määreellä. Ruotsi ei saanut yhtään \"konna\"
 -määritelmää, mutta sai sankarimääritelmän useissa kategorioissa. Sankarimääritelmän Ruotsi sai muun muassa innovaatiot-, tutkimus- ja kehitys-, taitojen parantaminen sekä luonnonympäristökategorioissa. Kaiken kai
kkiaan reformien toteuttamisen arvioinnin mukaan Ruotsi oli toiseksi edistynein Tanskan jälkeen.\nRuotsi keskittyy kasvustrategiassaan innovaatiopolitiikan lisäksi muun muassa julkisen talouden säästöihin, matalan 
inflaation säilyttämiseen, tasa-arvoiseen palkkaukseen naisten ja miesten välillä sekä työllisyyden vaalimiseen. Uuden porvarihallituksen myötä viimeksi mainitut työllisyysponnistelut tullevat olemaan keskeisiä muu
n muassa yrittäjyyden edellytysten parantamisen sekä veropolitiikan kautta.\nRuotsin suurpanostus innovatiivisiin tuotteisiin: PIE-p -ohjelma\nRuotsalainen suurpanostus innovatiivisiin tuotteisiin on PIE-p -ohjelma
, Product Innovation Engineering Program, jonka tarkoituksena on parantaa ruotsalaista tuotekehitystä ja liike-elämän kilpailukykyä."""
    paragraphs = [p_1_minor_deletions,p_2_major_deletions,p_3_perfect,p_4_different_doc]
    for i, p in enumerate(paragraphs,start=1):
        res = token_fuzzy_find(test_text, p)
        print(i,res)
        
