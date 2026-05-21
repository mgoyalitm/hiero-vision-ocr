def get_character_id(character: str) -> int:
    match character:
        case '1' | 'l' | 'I' | 'i':
            return 1
        case '2':
            return 2
        case '3':
            return 3
        case '4':
            return 4
        case '5':
            return 5
        case '6':
            return 6
        case '7':
            return 7
        case '8' | 'B':
            return 8
        case '9':
            return 9
        case '0' | 'O' | 'o' | 'D':
            return 10
        case 'A':
            return 11
        case 'a':
            return 12
        case 'b':
            return 13
        case 'C' | 'c':
            return 14
        case 'd':
            return 15
        case 'e':
            return 16
        case 'E':
            return 17
        case 'f':
            return 18
        case 'F':
            return 19
        case 'g':
            return 20
        case 'G':
            return 21
        case 'H':
            return 22
        case 'h':
            return 23
        case 'j' | 'J':
            return 24
        case 'k' | 'K':
            return 25
        case 'L':
            return 26
        case 'm':
            return 27
        case 'M':
            return 28
        case 'n':
            return 29
        case 'N':
            return 30
        case 'P' | 'p':
            return 31
        case 'q':
            return 32
        case 'Q':
            return 33
        case 'R':
            return 34
        case 'r':
            return 35
        case 'S' | 's':
            return 36
        case 't':
            return 37
        case 'T':
            return 38
        case 'U' | 'u':
            return 39
        case 'V' | 'v':
            return 40
        case 'W' | 'w':
            return 41
        case 'X' | 'x':
            return 42
        case 'Y' | 'y':
            return 43
        case 'Z' | 'z':
            return 44
        case _:
            return 45
        
def get_characters_from_id(i: int) -> list[str]:
    match i:
        case 1:
            return ['1','l','I','i']
        case 2:
            return ['2']
        case 3:
            return ['3']
        case 4:
            return ['4']
        case 5:
            return ['5']
        case 6:
            return ['6']
        case 7:
            return ['7']
        case 8:
            return ['8','B']
        case 9:
            return ['9']
        case 10:
            return ['0','O','o','D']
        case 11:
            return ['A']
        case 12:
            return ['a']
        case 13:
            return ['b']
        case 14:
            return ['C','c']
        case 15:
            return ['d']
        case 16:
            return ['e']
        case 17:
            return ['E']
        case 18:
            return ['f']
        case 19:
            return ['F']
        case 20:
            return ['g']
        case 21:
            return ['G']
        case 22:
            return ['H']
        case 23:
            return ['h']
        case 24:
            return ['j','J']
        case 25:
            return ['k','K']
        case 26:
            return ['L']
        case 27:
            return ['m']
        case 28:
            return ['M']
        case 29:
            return ['n']
        case 30:
            return ['N']
        case 31:
            return ['P','p']
        case 32:
            return ['q']
        case 33:
            return ['Q']
        case 34:
            return ['R']
        case 35:
            return ['r']
        case 36:
            return ['S','s']
        case 37:
            return ['t']
        case 38:
            return ['T']
        case 39:
            return ['U','u']
        case 40:
            return ['V','v']
        case 41:
            return ['W','w']
        case 42:
            return ['X','x']
        case 43:
            return ['Y','y']
        case 44:
            return ['Z','z']
        case _:
            return []
    
def get_directory_name(character: str) -> str:
    if character.isdigit():
        return f"digit_{character}"
    elif character.isupper():
        return f"upper_{character}"
    elif character.islower():
        return f"lower_{character}"
    else:
        raise ValueError(f"Unsupported character: {character}")