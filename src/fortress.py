import numpy as np

#localsearch1
def rowtopreshape1(row):
    if row == 0:
        rowtop = row + 0
    elif row == 1:
        rowtop = row - 0
    elif row ==6:
        rowtop = row - 3
    elif row ==5:
        rowtop = row -2
    else:
        rowtop = row - 1
        
    return rowtop

def rowbotreshape1(row):
    if row == 0:
        rowbot = row + 3
    elif row == 1:
         rowbot = row + 3
    elif row == 6:
         rowbot = row + 0
    elif row == 5:
        rowbot = row + 1
    else:
        rowbot = row +2
    
    return rowbot

def columnfrontreshape1(column):
    if column == 8:
        columnfront = column - 3
    elif column == 7:
        columnfront = column - 2
    elif column == 1:
        columnfront = column - 0
    elif column == 0:
        columnfront = column + 0
    else:
        columnfront = column -1
        
    return columnfront

def columnbackreshape1(column):
    if column == 8:
        columnback = column + 0
    elif column == 7:
        columnback = column + 1
    elif column == 1:
        columnback = column + 3
    elif column == 0:
        columnback = column + 3
    else:
        columnback = column + 2

    return columnback

#rowtop1 = rowtopreshape1(row)
#rowbot1 = rowbotreshape1(row)

#colfront1 = columnfrontreshape1(column)
#colback1 = columnbackreshape1(column)


#localsearch2
def rowtopreshape2(row):
    if row == 0:
        rowtop = row + 0
    elif row == 1:
        rowtop = row - 1
    elif row ==6:
        rowtop = row - 4
    elif row ==5:
        rowtop = row -3
    else:
        rowtop = row - 2
        
    return rowtop

def rowbotreshape2(row):
    if row == 0:
        rowbot = row + 5
    elif row == 1:
         rowbot = row + 4
    elif row == 6:
         rowbot = row + 1
    elif row == 5:
        rowbot = row + 2
    else:
        rowbot = row +3
    
    return rowbot

def columnfrontreshape2(column):
    if column == 8:
        columnfront = column - 4
    elif column == 7:
        columnfront = column - 3
    elif column == 1:
        columnfront = column - 1
    elif column == 0:
        columnfront = column + 0
    else:
        columnfront = column -2
        
    return columnfront

def columnbackreshape2(column):
    if column == 8:
        columnback = column + 1
    elif column == 7:
        columnback = column + 2
    elif column == 1:
        columnback = column + 4
    elif column == 0:
        columnback = column + 5
    else:
        columnback = column + 3

    return columnback


#rowtop = rowtopreshape2(row)
#rowbot = rowbotreshape2(row)

#colfront = columnfrontreshape2(column)
#colback = columnbackreshape2(column)