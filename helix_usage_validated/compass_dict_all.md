# Compass Dictionary (across GPT-2, Phi-3, Gemma-2)

Decoded via W_U @ V^T_k with per-row centering of W_U.
Top tokens along +v/-v for the top 4 SV axes of each passing compass head.

Axes per head: at least 4, extended until cumulative σ² ≥ 0.9, capped at 8.

| Model | Head | SV | σ | cum σ² | +v direction (top-8) | −v direction (top-8) |
|---|---|---:|---:|---:|---|---|
| gpt2 | L10H9 | 0 | 9.15 | 0.08 | His, his, He, his, His, he, He, himself | her, she, She, herself, she, She, Her, hers |
| gpt2 | L10H9 | 1 | 7.87 | 0.13 | her, she, herself, She, she, She, SHE, haar | their, Their, Their, they, They, their, They, THEIR |
| gpt2 | L10H9 | 2 | 7.27 | 0.18 | She, herself, her, she, she, hers, She, Her | Its, Its, its, ITS, itself, imo, it, It |
| gpt2 | L10H9 | 3 | 6.80 | 0.23 | Heard, ours, andowski, CVE, Kraken, Commentary, written, Released | eteen, your, enture, phan, approximately, >(, ific, inda |
| gpt2 | L10H9 | 4 | 5.99 | 0.26 | Its, Its, its, ITS, hers, itself, her, its | ourselves, yourselves, Our, our, Our, We, we, We |
| gpt2 | L10H9 | 5 | 5.84 | 0.29 | your, yourself, your, Your, Your, YOUR, yours, you | ourselves, our, ours, we, Our, We, us, We |
| gpt2 | L10H9 | 6 | 5.49 | 0.32 | my, myself, me, my, My, My, I, I | yourselves, their, THEIR, they, they, THEY, Their, They |
| gpt2 | L10H9 | 7 | 5.36 | 0.34 | iating, distributing, icking, thening, noticing, producing, converting, inflicting | rued, my, oled, igrated, idated, oided, myself, ursed |
| gpt2 | L9H7 | 0 | 9.23 | 0.05 | their, Their, Their, their, they, THEIR, They, themselves | his, He, his, he, His, He, him, His |
| gpt2 | L9H7 | 1 | 8.87 | 0.10 | herself, her, She, she, Her, she, She, Her | His, his, his, His, He, He, he, himself |
| gpt2 | L9H7 | 2 | 8.46 | 0.14 | Its, its, Its, ITS, itself, its, it, It | He, His, himself, his, herself, his, His, He |
| gpt2 | L9H7 | 3 | 7.69 | 0.18 | humble, SIL, Silence, Low, Less, lesser, spared, Idle | xon, ワン, warts, arching, major, istries, anos, VIDIA |
| gpt2 | L9H7 | 4 | 7.20 | 0.21 | chance, laughter, improvis, Optional, invention, �, options, rez | rigid, strict, areth, Serious, Rig, odan, entrenched, sting |
| gpt2 | L9H7 | 5 | 6.98 | 0.24 | Both, Both, Neither, Neither, neither, Either, both, Either | others, Others, god, rodu, sequence, Nine, enegger, doi |
| gpt2 | L9H7 | 6 | 6.74 | 0.27 | smallest, iny, tiny, PsyNetMessage, simplicity, humble, Small, passively | ocious, Highly, azeera, itbart, iannopoulos, Sands, qus, Both |
| gpt2 | L9H7 | 7 | 6.58 | 0.29 | assetsadobe, plain, wal, both, ============, blogspot, oute, cow | mented, ql, cious, Carter, chell, Castro, ische, earcher |
| gpt2 | L11H1 | 0 | 18.26 | 0.06 | adem, uman, resy, essen, apeake, illin, aku, xual | widow, Scand, niece, Huma, daughter, estranged, Guarant, actress |
| gpt2 | L11H1 | 1 | 13.35 | 0.09 | men, MEN, Guys, gentlemen, guys, Men, boys, Men | Alicia, actress, Jenny, Nikki, Janet, Elizabeth, Donna, Denise |
| gpt2 | L11H1 | 2 | 12.99 | 0.11 | kid, child, Parent, parent, Parents, youngster, parents, child | women, ladies, Lady, women, lady, Lady, Women, Ladies |
| gpt2 | L11H1 | 3 | 12.71 | 0.14 | wives, Women, women, women, Mother, wife, Women, mothers | student, student, Student, pupil, Student, boy, Students, Bieber |
| gpt2 | L11H1 | 4 | 12.53 | 0.17 | Male, riott, Persons, man, Edward, partner, Actor, Director | Queen, Girl, girl, queen, Duchess, vana, girls, Queen |
| gpt2 | L11H1 | 5 | 12.50 | 0.19 | woman, Woman, woman, Goddess, pmwiki, Woman, Ms, Girl | brothers, brother, party, brother, parties, nephew, cipled, apers |
| gpt2 | L11H1 | 6 | 11.89 | 0.22 | Person, persons, Persons, stocks, person, chin, maxwell, coron | boys, girls, mascul, guys, men, Men, sisters, Girls |
| gpt2 | L11H1 | 7 | 11.88 | 0.24 | Her, Her, Leader, She, Him, Daughter, femin, HER | cery, Bill, white, ports, Bill, executive, adan, Esk |
| phi3 | L28H1 | 0 | 13.67 | 0.09 | his, his, himself, lui, him, him, he, His | her, she, hers, herself, she, elle, She, ella |
| phi3 | L28H1 | 1 | 10.04 | 0.13 | his, him, himself, his, him, lui, he, ihm | Their, their, loro, leurs, themselves, Its, onymes, they |
| phi3 | L28H1 | 2 | 9.27 | 0.17 | our, ourselves, Our, our, notre, nous, ours, OUR | my, my, mys, My, My, myself, MY, MY |
| phi3 | L28H1 | 3 | 7.61 | 0.20 | your, Your, yourself, yours, your, Your, selves, ва | my, my, myself, mys, My, My, MY, our |
| phi3 | L28H1 | 4 | 7.33 | 0.23 | Mr, Mr, Mrs, Monsieur, Dr, Dr, Prof, Professor | iten, ible, Menu, quet, liches, succeed, mouth, pn |
| phi3 | L28H1 | 5 | 7.06 | 0.25 | their, Their, themselves, loro, leur, leurs, ihre, их | Mr, Mr, yourself, ones, your, my, myself, Mrs |
| phi3 | L28H1 | 6 | 6.01 | 0.27 | Dr, Dr, dr, dr, Doctor, DR, drum, doctor | your, your, Your, Your, yourself, votre, yours, selves |
| phi3 | L28H1 | 7 | 5.81 | 0.28 | Dr, Dr, dr, dr, your, your, Your, Your | Mr, Mr, Mrs, mines, osen, мож, agt, ella |
| phi3 | L27H16 | 0 | 11.21 | 0.06 | him, lui, his, him, ihm, his, ihn, himself | they, they, They, They, Their, their, she, them |
| phi3 | L27H16 | 1 | 10.27 | 0.11 | they, they, They, They, their, Their, loro, ihnen | she, hers, she, her, ella, herself, She, ela |
| phi3 | L27H16 | 2 | 9.82 | 0.15 | you, you, your, You, You, your, Your, yourself | she, she, hers, ella, her, She, elle, ela |
| phi3 | L27H16 | 3 | 8.67 | 0.19 | you, you, You, You, yours, your, vous, your | me, my, us, myself, mine, my, mys, I |
| phi3 | L27H16 | 4 | 7.42 | 0.21 | us, ourselves, we, our, nous, 我, we, him | Its, its, its, it, It, It, ts, itself |
| phi3 | L27H16 | 5 | 7.23 | 0.24 | ourselves, us, we, our, nous, Our, We, amos | me, myself, my, I, mys, mine, my, mine |
| phi3 | L27H16 | 6 | 6.42 | 0.26 | ioni, arod, POS, endif, getElement, Wasser, Prof, ->_ | us, you, our, you, yours, nous, 我, me |
| phi3 | L27H16 | 7 | 5.58 | 0.27 | him, me, you, us, you, ihm, lui, ihn | walt, oss, getMessage, utt, burgh, ğ, uropa, riz |
| phi3 | L24H10 | 0 | 9.42 | 0.10 | their, they, Their, loro, they, They, They, leur | Its, its, him, itself, suoi, its, його, jego |
| phi3 | L24H10 | 1 | 7.70 | 0.17 | its, Its, its, itself, Their, their, it, loro | his, him, himself, his, ihn, him, he, 他 |
| phi3 | L24H10 | 2 | 6.94 | 0.22 | she, she, elle, шла, ella, ela, her, She | him, his, his, lui, him, himself, ihm, ihn |
| phi3 | L24H10 | 3 | 6.60 | 0.27 | illé, Your, órico, ático, ljen, inha, utat, uesto | she, hers, her, she, herself, ella, She, elle |
| phi3 | L24H10 | 4 | 4.82 | 0.30 | company, �, country, Company, haar, >[, Company, company | yours, your, your, yourself, 你, suas, votre, sua |
| phi3 | L24H10 | 5 | 4.61 | 0.32 | your, dessen, yours, Your, Your, your, hers, respective | my, myself, my, mys, olg, UG, ems, atre |
| phi3 | L24H10 | 6 | 4.36 | 0.34 | my, our, ourselves, my, Our, myself, notre, mys | your, your, Your, Your, yours, yourself, votre, you |
| phi3 | L24H10 | 7 | 3.93 | 0.36 | my, them, ones, myself, my, mys, ones, its | together, ope, uset, andid, roe, us, ferrer, anu |
| phi3 | L26H22 | 0 | 9.43 | 0.07 | 该, dieses, this, this, 此, This, This, ese | these, These, estos, estas, them, diesen, questi, those |
| phi3 | L26H22 | 1 | 7.57 | 0.11 | him, ihn, he, him, celui, ce, ihm, lui | she, she, ella, She, Esta, questa, esta, ela |
| phi3 | L26H22 | 2 | 6.30 | 0.14 | it, she, ce, she, uen, It, èg, CE | this, this, these, dieser, dieses, diesen, diese, diesem |
| phi3 | L26H22 | 3 | 6.22 | 0.17 | these, she, this, diese, These, dieser, she, this | them, they, They, they, They, ihnen, ellos, ils |
| phi3 | L26H22 | 4 | 5.86 | 0.20 | it, those, Those, these, It, ese, celui, It | ones, they, They, they, them, They, ones, this |
| phi3 | L26H22 | 5 | 5.77 | 0.22 | this, this, it, dieses, This, This, questo, questa | ese, esa, quella, That, That, him, thats, those |
| phi3 | L26H22 | 6 | 5.02 | 0.24 | him, ella, ihm, ihn, it, lui, él, she | Such, such, such, uch, такой, тако, та, så |
| phi3 | L26H22 | 7 | 4.97 | 0.26 | those, Those, ose, ese, dieses, ceux, ient, ibm | him, she, ihn, him, lui, ihm, she, Such |
| phi3 | L25H24 | 0 | 6.46 | 0.04 | themselves, their, Their, leur, loro, leurs, ihre, ihren | yourself, ourselves, your, our, your, Your, yours, Your |
| phi3 | L25H24 | 1 | 5.62 | 0.08 | his, himself, his, him, him, seinem, he, His | Its, its, itself, its, шло, my, шла, ts |
| phi3 | L25H24 | 2 | 5.45 | 0.11 | herself, yourself, her, your, your, selves, she, Your | his, himself, his, seinem, seiner, seinen, seines, seine |
| phi3 | L25H24 | 3 | 5.42 | 0.14 | ourselves, myself, herself, her, my, our, hers, she | your, your, Your, Your, yourself, selves, its, votre |
| phi3 | L25H24 | 4 | 5.17 | 0.16 | yourself, ್, my, rache, myself, your, vě, __. | ;, ;", ;`, ;</, ;, .;, ;\, %; |
| phi3 | L25H24 | 5 | 4.93 | 0.19 | myself, my, yourself, mys, ourselves, your, my, Your | herself, her, hers, she, she, his, ella, himself |
| phi3 | L25H24 | 6 | 4.65 | 0.21 | {},, [],, %,, %,, '',, "",, _,, ?, | ."), ':', ()., ']., _., '., .:, . |
| phi3 | L25H24 | 7 | 4.43 | 0.23 | my, myself, my, mys, mine, My, My, mine | ourselves, our, Our, notre, our, nous, we, OUR |
| gemma | L21H4 | 0 | 2.35 | 0.10 | him, 彼は, he, his, ihn, 彼が, 彼の, ньому | she, her, she, She, herself, 그녀, 彼女の, shes |
| gemma | L21H4 | 1 | 2.07 | 0.17 | him, she, he, his, onun, her, ihm, He | they, They, They, they, mereka, them, Mereka, Mereka |
| gemma | L21H4 | 2 | 1.98 | 0.24 | him, he, his, egli, 彼は, beliau, her, 彼が | its, Its, Оно, Its, оно, its, 它, которое |
| gemma | L21H4 | 3 | 1.52 | 0.28 | RectangleBorder, تضيفلها, ajiban, Normdatei, haraan, myſelf, JspWriter, you | she, him, his, her, he, their, they, Their |
| gemma | L21H4 | 4 | 1.17 | 0.30 | their, Their, Their, 彼らの, THEIR, 彼らは, their, leur | ones, TagHelper, your, you, those, your, youre, those |
| gemma | L21H4 | 5 | 1.14 | 0.32 | vocês, Their, vuestro, endif, 彼らの, InitVars, TagMode, berdua | its, Its, Its, 它們, its, ones, 它们, celles |
| gemma | L21H4 | 6 | 1.03 | 0.34 | you, your, your, you, yourself, Your, youre, You | we, my, our, We, We, kami, нами, 我們 |
| gemma | L21H4 | 7 | 0.93 | 0.35 | elles, ellas, Elles, Elles, girls, ISupport, fjspx, sisters | suoi, its, quelli, /"), $"), "), '"), ceux |