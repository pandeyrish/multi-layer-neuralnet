
def mini_geant(num_experiments,min_start_difference,max_start_difference,min_size,max_size,min_velocity,max_velocity,output_folder):
    #from PIL import Image
    import os
    try:
        os.mkdir(output_folder)
    except:
        pass
    from PIL import Image, ImageDraw, ImageFilter
    import numpy as np
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    img.save("blankpage2.jpg", "JPEG")
    for fdf in range(1, num_experiments):
        import numpy as np
        import random

        my_50x50_matrix_at_runtime = np.zeros((64, 64))

        import random

        # min_start_difference = 1
        # max_start_difference = 5
        gun1start = random.randint(min_start_difference, max_start_difference)
        gun2start = random.randint(min_start_difference, max_start_difference)

        # min_size = 1
        # max_size = 5
        gun1size = random.randint(min_size, max_size)
        gun2size = random.randint(min_size, max_size)

        # min_velocity = 1
        # max_velocity = 5
        gun1velocity = random.randint(min_velocity, max_velocity)
        gun2velocity = random.randint(min_velocity, max_velocity)

        collisiontype = random.randint(1, 3)

        setlocation = 0
        setlocation1 = 0
        newsize = 0
        newsize1 = 0
        INELASTIC = 1
        ELASTIC = 2
        lr = 0
        INELASTIC_DECAY = 3
        INELASTIC_MULTIPLE = 4
        # pointer = setlocation
        for i in range(1, 64 + 1):

            if newsize != 0:
                velocity = int(velocity)
                array = my_50x50_matrix_at_runtime[-i]
                if gun1velocity * gun1size >= gun2velocity * gun2size:
                    for v in range(0, newsize):
                        if setlocation + v <= (len(array) - 1):
                            array[setlocation + v] = 3
                    for location in range(0, velocity):
                        setlocation = setlocation + 1

                else:
                    for t in range(0, newsize):
                        if setlocation1 + t <= (len(array) - 1):
                            array[-(setlocation1 + t + 1)] = 3
                    for location1 in range(0, velocity):
                        setlocation1 = setlocation1 + 1


            elif newsize1 != 0:
                if time != 0:
                    time = time - 1
                    velocity = int(velocity)
                    array = my_50x50_matrix_at_runtime[-i]
                    if gun1velocity * gun1size >= gun2velocity * gun2size:
                        for v in range(0, newsize1):
                            if setlocation + v <= (len(array) - 1):
                                array[setlocation + v] = 3
                        for location in range(0, velocity):
                            setlocation = setlocation + 1
                            setlocation1 = setlocation1 - 1
                    else:
                        for t in range(0, newsize1):
                            if setlocation1 + t <= (len(array) - 1):
                                array[-(setlocation1 + t + 1)] = 3
                        for location1 in range(0, velocity):
                            setlocation1 = setlocation1 + 1
                            setlocation = setlocation - 1
                else:

                    newsize1 = 0

            else:

                array = my_50x50_matrix_at_runtime[-i]

                for v in range(0, gun1size):
                    if setlocation + v <= (len(array) - 1):

                        if int(array[setlocation + v]) == 2:
                            if gun1start == gun2start:
                                if collisiontype == INELASTIC:
                                    newsize = int(gun1size + gun2size)
                                    velocity = int(gun1velocity * gun1size + gun2velocity * gun2size) / (
                                                gun1size + gun2size)

                                    break
                                if collisiontype == ELASTIC:
                                    gun1splocity = (gun2size * gun2velocity) / gun1size
                                    gun2splocity = (gun1size * gun1velocity) / gun2size
                                    gun1velocity = gun1splocity
                                    gun2velocity = gun2splocity
                                    lr = 1
                                    break
                                if collisiontype == INELASTIC_DECAY:
                                    newsize1 = int(gun1size + gun2size)
                                    velocity = int(gun1velocity * gun1size + gun2velocity * gun2size) / (
                                                gun1size + gun2size)
                                    time = random.randint(min_start_difference, max_start_difference)
                                    break
                            else:
                                array[setlocation + v] = 1
                        else:
                            # print(int(array[-(setlocation1+v+1)]))
                            array[setlocation + v] = 1

                for t in range(0, gun2size):
                    if setlocation1 + t <= (len(array) - 1):
                        if int(array[-(setlocation1 + t + 1)]) == 1:
                            if gun1start == gun2start:
                                if collisiontype == INELASTIC:
                                    newsize = int(gun1size + gun2size)
                                    velocity = int(gun1velocity * gun1size + gun2velocity * gun2size) / (
                                                gun1size + gun2size)
                                    break

                                if collisiontype == INELASTIC_DECAY:
                                    newsize1 = int(gun1size + gun2size)
                                    velocity = int(gun1velocity * gun1size + gun2velocity * gun2size) / (
                                                gun1size + gun2size)
                                    time = random.randint(min_start_difference, max_start_difference)
                                    break

                                if collisiontype == ELASTIC:

                                    if lr == 0:
                                        gun1splocity = (gun2size * gun2velocity) / gun1size
                                        gun2splocity = (gun1size * gun1velocity) / gun2size
                                        gun1velocity = gun1splocity
                                        gun2velocity = gun2splocity
                                        break

                                    else:
                                        lr = 0
                                        break

                            else:
                                array[-(setlocation1 + t + 1)] = 2
                        else:
                            array[-(setlocation1 + t + 1)] = 2

                gun1velocity = int(gun1velocity)
                gun2velocity = int(gun2velocity)
                for location in range(0, gun1velocity):
                    setlocation = setlocation + 1
                for location1 in range(0, gun2velocity):
                    setlocation1 = setlocation1 + 1

        import PIL.ImageDraw as ImageDraw, PIL.Image as Image, PIL.ImageShow as ImageShow

        if gun1start == gun2start:
            if collisiontype == 1:
                collisiontype = "INELASTIC"
            if collisiontype == 2:
                collisiontype = "ELASTIC"
            if collisiontype == 3:
                collisiontype = "INELASTIC_DECAY"
            if collisiontype == 4:
                collisiontype = "INELASTIC_MULTIPLE"
        else:
            collisiontype = "NO_COLLISION"
        img = Image.open("blankpage2.jpg")
        # img = Image.new("RGB", (1000,1000))
        draw = ImageDraw.Draw(img)
        for y in range(0, 64):
            for x in range(0, 64):
                if my_50x50_matrix_at_runtime[x][y] != 0:
                    draw.point((y, x), fill=0)
        img.save(output_folder+"/"+str(fdf) + "_" + collisiontype + ".png")


import sys
i = 0
for arg in sys.argv:

	globals()["arg" + str(i)] = arg
	i = i+1

num_experiments = int(arg1)
min_start_difference = int(arg2)
max_start_difference =  int(arg3)
min_size = int(arg4)
max_size = int(arg5)
min_velocity = int(arg6)
max_velocity = int(arg7)
output_folder = arg8

mini_geant(num_experiments,min_start_difference,max_start_difference,min_size,max_size,min_velocity,max_velocity,output_folder)