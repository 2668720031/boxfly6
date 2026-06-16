from drone import TelloPy

drone = TelloPy()

drone.connect()
drone.land()

# drone.get_all_funcs_desc()