import sys
import os
import binascii
import struct
import numpy as np
import itertools

#Change the listed path here to read from different bin files instead.
binfile_path = '../../data/raw/FLS_Stream/Oculus_20200304_161016.bin'
byte_counter = 0

valid_itemheader_hex = 'ddccbbaa'


#### DEFINITION of BluePrint Subsea's Oculus FLS Sonar Ping Objects (using Oculus M750D sensor in this case) ####
class OculusFLSPing:

    def __init__(self, pingConfig=None, masterMode=0, pingRate_type=b'\x01', gammaCorrection_usr=0, usr_flags=0, range_usr=20, gainPercent_usr=10.0, \
                 soundVel_usr=0.0, salinity_usr=0.0, pingId=0, freq=0.0, temp=0.0, pressure=0.0, heading=0.0, pitch=0.0, roll=0.0, soundVel_act=0.0, pingStart=0, dataSize_id=0, rangeRes=0.0, \
                 N=0, M=0, imgOffset=0, imgSize=0, msgSize=0, verbose=False):

        self.pingRate_dict = {b'\x00': 10.00, b'\x01': 15.00, b'\x02': 40.00, b'\x03': 5.00, b'\x04': 2.00, b'\x05': None}

        if pingConfig is None:
            self.pingConfig =  {'master_mode'           : masterMode,
                                'pingRate_bytes'        : pingRate_type, #self.pingRate_dict[pingRate_type],
                                'user-gamma_correction' : gammaCorrection_usr,
                                'user-set_flags'        : usr_flags,
                                'user-stop_range'       : range_usr,
                                'user-gainPercent'      : gainPercent_usr,
                                'user-soundVel'         : soundVel_usr,
                                'user-salinity'         : salinity_usr,
                                'pingId'                : pingId,
                                'beam_frequency'        : freq,
                                'water_temperature'     : temp,
                                'water_pressure'        : pressure,
                                'sensor_heading'        : heading,
                                'sensor_pitch'          : pitch,
                                'sensor_roll'           : roll,       
                                'soundVel_actual'       : soundVel_act,
                                'pingStart_time'        : pingStart,
                                'dataSize_bytes'        : dataSize_id + 1,
                                'range_resolution'      : rangeRes,
                                'N'                     : N,
                                'M'                     : M,
                                'brgTable_offset'       : 122, # Magic number of the Bearing Table offset, based on how it's logged into the raw payload data & 
                                                               # information shared from DataStructure document. Need to be fixed in future versions, such 
                                                               # that the UWR logger also logs in the raw Bearing Table bytes separately, or also logs in the 
                                                               # offset as extra info.
                                'image_offset'          : imgOffset,
                                'image_size'            : imgSize,
                                'message_size'          : msgSize,
                                #'payload_dtype'         : 
                                }
        else:
            self.pingConfig = pingConfig

        self.usr_flags = self.pingConfig['user-set_flags']
        self.flag_info = {'as_meters'   : True,
                          'dtype_size'  : None,
                          'send_again'  : True,
                          'return_type' : 'simple'}
        if verbose:
            print('User set flags for this ping:', usr_flags)
        #self.usr_flags_int = int.from_bytes(usr_flags, byteorder='little')
        self.parse_flag_info()
        if verbose:
            print(self.flag_info)
        
        self.dtype_dict = {1:np.uint8, 2:np.uint16, 3:np.uint32}
        self.data_bytesize = self.pingConfig['dataSize_bytes']
        self.acousticImg_raw = bytearray(b'')
        self.acousticImg_pix = np.array([])
        self.acousticImg_matrix = None
        self.brgTable_raw = None
        self.brgTable_val = None
        self.echoData = None

    def parse_flag_info(self):
        usr_flags = self.usr_flags
        usr_flags_int = int.from_bytes(usr_flags, byteorder='little', signed=False)
        flag_bit_0 = int.from_bytes(b'\x80', byteorder='little')
        flag_bit_1 = int.from_bytes(b'\x40', byteorder='little')
        flag_bit_2 = int.from_bytes(b'\x20', byteorder='little')
        flag_bit_3 = int.from_bytes(b'\x10', byteorder='little')
        #print("Flags set by user:", type(usr_flags), usr_flags, int.from_bytes(usr_flags, byteorder='little', signed=False))
        #check each of the user flags' bits
        #First, check bit 0
        if (usr_flags_int & flag_bit_0).to_bytes(1, byteorder='little') == b'\x80':
            self.flag_info['as_meters'] = True
        else:
            self.flag_info['as_meters'] = False

        if (usr_flags_int & flag_bit_1).to_bytes(1, byteorder='little') == b'\x40':
            self.flag_info['dtype_size'] = 2
        else:
            self.flag_info['dtype_size'] = 1

        if (usr_flags_int & flag_bit_2).to_bytes(1, byteorder='little') == b'\x20':
            self.flag_info['send_again'] = True
        else:
            self.flag_info['send_again'] = False
        
        if (usr_flags_int & flag_bit_3).to_bytes(1, byteorder='little') == b'\x10':
            self.flag_info['return_type'] = 'simple'
        else:
            self.flag_info['return_type'] = 'full'
        #print(usr_flags, self.flag_info)

    def cvt_acousticImg_byte2pixels(self, verbose=True):
        dtype_size = self.flag_info['dtype_size']
        acousticImg_datatype = self.dtype_dict[self.flag_info['dtype_size']]
        max_M = self.pingConfig['M']
        max_N = self.pingConfig['N']
        max_pixels = max_M * max_N // dtype_size
        #print(max_pixels)
        
        if len(self.acousticImg_raw) > 0:
            print(len(self.acousticImg_raw), max_pixels, acousticImg_datatype, int.from_bytes(self.usr_flags, byteorder='little', signed=False))
            print(self.usr_flags)
       
            pixel_data = np.frombuffer(self.acousticImg_raw, dtype=acousticImg_datatype, count=int(max_pixels) )
            print(pixel_data.shape)
            #if(acousticImg_datatype!=np.uint8):
            if acousticImg_datatype != np.uint8:
            	pixel_data = pixel_data.view(np.uint8)  # Reinterpret bytes as uint8

            	
            if verbose:
                print("raw bytes for acoustic image:")
                print(self.acousticImg_raw, pixel_data, len(pixel_data))
            self.acousticImg_pix = np.append(self.acousticImg_pix, pixel_data)
        else:
            if verbose:
                print("Raw acoustic image doesn't exist yet! Please ensure it's been created before invoking this conversion function!")
                return False
        
        return True 

    def create_acousticImg_matrix(self, verbose=False):
        M = self.pingConfig['M']
        N = self.pingConfig['N']

        if self.acousticImg_pix is not None:
            if verbose:
                print(len(self.acousticImg_pix))
            self.acousticImg_matrix = np.reshape(self.acousticImg_pix, (M, N))
        else:
            return

    def update_brgTable_raw(self, payload_data, brgTable_offset=None, verbose=False):
        '''
        Function to obtain the bearing table of the current ping.

        Arguments:
        - payload_data => Payload data bytes that was received as part of the current sonar ping packet.
        - verbose => (Optional) Flag to view the debugging prints or not
        '''
        
        nBeams = self.pingConfig['N']

        if brgTable_offset is None:
            brgTable_offset = self.pingConfig['brgTable_offset']

        if self.brgTable_raw is None:
            self.brgTable_raw = payload_data[brgTable_offset : (brgTable_offset + 2 * int(nBeams))] # 2 is size of 'short' used in the CPP Logger for brgTable. 
                                                                                                    # Need to be fixed in future versions such that this type of 
                                                                                                    # magic variable doesn't get used too often.
            if verbose:
                print("Raw Bearing Table:", self.brgTable_raw)
            self.create_brgTable_val(verbose=verbose)
        else:
            if verbose:
                print("Bearing Table already exists for this ping!")
            return False
        return True

    def create_brgTable_val(self, verbose=False):
        '''
        Function to get the actual (floating point) values from the raw Bearing Table (if available) which are in bytes.
        '''
        nBeams = self.pingConfig['N']
        brgTable_datatype = np.int16 # Magic variable. Obtained from knowledge that the original Bearing  Table was logged as a series of *short* data types in 
                                     # the CPP logger. Need to be fixed in future version such that logger also logs this information separately.
        if self.brgTable_val is None:
            if self.brgTable_raw is None:
                if verbose:
                    print("Raw Bearing Table doesn't exist yet!")
                return False
            else:
                self.brgTable_val = np.frombuffer(self.brgTable_raw, dtype=brgTable_datatype, count=int(nBeams)) / 100.0
                if verbose:
                    print("Value Bearing table:", self.brgTable_val)
                return True
        else:
            if verbose:
                print("Bearing Table values for this ping already exists!")
            return False

    def update_acousticImg_raw(self, payload_data, img_offset=None, verbose=False):
        '''
        #!!NOTE_: Current recommended version to use. This one appends water column data assuming it's being stored 
        #as bytestring data!!

        Function to update/append the ping's water column data with the new incoming data bytes.
        Note that the water colum matrix is only created if the water column data list (in here named 
        self.water_column_pixels) is full.
        
        Arguments:
        - payload_data => Payload data bytes that was received as part of the current sonar ping packet. 
        - img_offset => Offset for current payload_data to get the actual image bytes. To then be used for updating self.waterColumn_raw
        - verbose => (Optional) Flag to view the debugging prints or not.

        Returns:
        - flag => Indicating whether updating of the raw waterColumn data was successful or not
        '''
        dtype_size = self.flag_info['dtype_size']#self.data_bytesize
        max_M = self.pingConfig['M']
        max_N = self.pingConfig['N']
        max_byte_size = max_M * max_N * dtype_size

        if img_offset is None:
            img_offset = self.pingConfig['image_offset']
        if verbose:
            print("image offset", img_offset)
        img_bytes = payload_data[img_offset: ]
        #print(len(img_bytes), max_byte_size)
        if len(img_bytes) <= max_byte_size:
            try:
                self.acousticImg_raw = img_bytes
                #print(img_bytes)
                self.cvt_acousticImg_byte2pixels()
                
                self.create_acousticImg_matrix(verbose=verbose)
            except Exception as exc:
                print("Cannot create acoustic Image from data. Ensure image data and offset are correctly marked in payload!", len(img_bytes))
                print(exc)
                return False
            return True
        else:
            if verbose:
                print("The current size of acoustic Image data exceeds the reported size in the Ping's config. \
                        \n Please ensure that the data's size is accurate!")
            return False

    def get_echoData(self, maxIntensity, verbose=False):
        '''
        Function to calculate the echo values for each ping's pixels.
        '''
        
        gain = self.pingConfig['user-gainPercent']
        echo_vals = self.acousticImg_pix / gain
        if verbose:
            print("Echo data:", echo_vals.dtype, echo_vals, gain)
        self.echoData = np.maximum(0, np.minimum(echo_vals/maxIntensity, 1.0))
        return self.echoData
    
    def get_pingConfig(self):
        return self.pingConfig
    
    def get_brgTable_val(self):
        return self.brgTable_val
    
    def get_acousticImg_raw(self):
        return self.acousticImg_raw
    
    def get_acousticImg_pix(self):
        return self.acousticImg_pix

    def get_acousticImg_matrix(self):
        return self.acousticImg_matrix

############################ END OF CLASS DEFINITIONS #####################################

#short helper function to help chunk data into specified sizes            
def chunk_databytes(size, data):
    for i in range(0, len(data), size):
        yield data[i:i+size]


#function to convert bytes into integers)
def get_int_data(byte_data, signed, byteorder='big'):
    '''
    Function to Convert bytes into integers, and should be able to automatically read and figure out type of integer that is being represented by the bytes
    (i.e. the function should be able to tell whether the integer is of type int16/uint16 (2 bytes data), int32/uint32 (4 bytes data), or int64/uint64 (8 bytes data))

    Arguments:
    - byte_data => Data bytes to be converted into integer
    - signed => Determine whether the bytes is representing a signed, or unsigned version of the integer (i.e. intX, or uintX type)

    Returns: Data, represented with converted Integer type
    '''
    data_list = list(byte_data)
    data_list.reverse()
    return int.from_bytes(bytes(data_list), byteorder=byteorder, signed=signed)

def read_from_file(bin_file, size, data_type, signed=False):
    '''
    Function to help automate the conversion of bytes from a binary file to the specified data type.
    Note: Needs a file pointer to exist first, supplied by an open() function performed on the file.

    Arguments:
    - bin_file => pointer to a currently opened binary file
    - size => size of data, in bytes
    - data_type => type of data for the data bytes to be converted to
    - signed => Determine whether the data type is of unsigned or signed variety

    Returns: Variable/Data converted to specified data_type from binary data of specified size extracted from bin_file.
    '''
    global byte_counter
    data_byte = bin_file.read(size)
    byte_counter += size
    if data_type == 'int':
        return get_int_data(data_byte, signed)
    elif data_type == 'float32':
        return struct.unpack('f', data_byte)[0]
    elif data_type == 'float64':
        return struct.unpack('d', data_byte)[0]


def parse_PingPacket(bin_file, is_oculus_ext, verbose=False):
    '''
    Function to parse the custom UWR 'header' bytes for the sonar data packet stored in the binary file.
    Note that the UWR Log Header uses a different struct compared to the original Oculus Log Header.

    Arguments:
    - binfile => Object pointer representing an opened binary file (assumes the use of open() function previously)
    - is_oculus_ext => Flag to indicate whether file is using an .oculus extension or not
    - verbose => Verbosity of the function for debugging purposes.

    Returns: timestamp of packet, size of packet, packet_validity 
    '''
    packet_header = bin_file.read(4)
    header_size = bin_file.read(4)
    data_type = bin_file.read(2)
    data_version = bin_file.read(2)
    #if not is_oculus_ext:
    buffer = bin_file.read(4) #For an unknown reason, it seems that the Qt IO Writing introduced a weird '\x00'(hex form) bytes each time
                                #a double data type is written. (look at line 564 in Oculus_PingGenerator function for other case)
    timestamp_bytes = bin_file.read(8)
    if is_oculus_ext:
        compression_bytes = bin_file.read(4)
    configsize_bytes = bin_file.read(4)
    payloadsize_bytes = bin_file.read(4)

    if verbose:
        print("Packet Header:", packet_header)
        print("Header size:", header_size) 
        print("Type of data:", data_type) 
        print("Version of data:", data_version)
        if is_oculus_ext:
            print("Compression of data:", compression_bytes)
    
    if packet_header.hex() != 'ddccbbaa': #b'\xaabbccdd': #sanity check for UWR's header
        if verbose:
            print('Header received:', packet_header.hex())
            print("Invalid packet header received!")
        return None, 0, 0, None, None, False
    else:
        packet_timestamp =  struct.unpack('d', timestamp_bytes)[0] 
        payload_size = int.from_bytes(payloadsize_bytes, byteorder='little', signed=False)
        config_size = int.from_bytes(configsize_bytes, byteorder='little', signed=False)
        if not is_oculus_ext:
            packet_config = bin_file.read(config_size)
            payload_data = bin_file.read(payload_size)
        else:
            packet_config = None
            original_size = config_size
            config_size = 0

            buffer = bin_file.read(4) #weird buffer that occurs between payload and packet information bytes

            payload_data = bin_file.read(payload_size)

        if verbose:
            print("Oculus Extension:", is_oculus_ext)
            print("Timestamp bytes:", timestamp_bytes, packet_timestamp) 
            print("Config size bytes:", configsize_bytes, config_size) 
            print("Payload size bytes:", payloadsize_bytes, payload_size)

    return packet_timestamp, (config_size + payload_size), payload_size, packet_config, payload_data, True

def parse_packet_UWRconfig(packet_config, verbose=False):
    '''
    [For use with '.bin' files]
    Function to parse configuration information of a sonar ping object, 
    if using the custom UWR binary data file format (files with extension '.bin')
    '''

    pingRate_dict = {b'\x00': 10.00, b'\x01': 15.00, b'\x02': 40.00, b'\x03': 5.00, b'\x04': 2.00, b'\x05': None}

    masterMode = get_int_data(packet_config[0:1], False, byteorder='little')
    pingRate_type = packet_config[1:2]

    gammaCorrection_bytes = packet_config[2:3]

    if gammaCorrection_bytes == b'\x00':
        gammaCorrection_usr = 1.0
    else:
        gammaCorrection_usr = get_int_data(packet_config[2:3], False, byteorder='little')/255. * 1.0

    usr_flags = packet_config[3:4]
    range_usr = struct.unpack('d', packet_config[8:16])[0] #[4:12])[0]
    gainPercent_usr = struct.unpack('d', packet_config[16:24])[0] #[12:20])[0]
    soundVel_usr = struct.unpack('d', packet_config[24:32])[0] #[20:28])[0]
    salinity_usr = struct.unpack('d', packet_config[32:40])[0] #[28:36])[0]
    pingId = int.from_bytes(packet_config[40:44], byteorder='little', signed=False) #int.from_bytes(packet_config[36:40], byteorder='little', signed=False)
    freq = struct.unpack('d', packet_config[48:56])[0] #[40:48])[0]
    temp = struct.unpack('d', packet_config[56:64])[0]#[48:56])[0]
    pressure = struct.unpack('d', packet_config[64:72])[0]#[56:64])[0]
    soundVel_act = struct.unpack('d', packet_config[72:80])[0] #[64:72])[0]
    pingStart = int.from_bytes(packet_config[80:84], byteorder='little', signed=False) #get_int_data(packet_config[64:68], False, byteorder='little')
    dataSize_id = int.from_bytes(packet_config[84:85], byteorder='little', signed=False) #get_int_data(packet_config[68:69], False, byteorder='little')
    rangeRes = struct.unpack('d', packet_config[88:96])[0] #[77:85])[0]
    M = int.from_bytes(packet_config[96:98], byteorder='little', signed=False)#int.from_bytes(packet_config[85:87], byteorder='little', signed=False)
    N = int.from_bytes(packet_config[98:100], byteorder='little', signed=False)#int.from_bytes(packet_config[87:89], byteorder='little', signed=False)
    imgOffset = int.from_bytes(packet_config[100:104], byteorder='little', signed=False) #int.from_bytes(packet_config[89:93], byteorder='little', signed=False)
    imgSize = int.from_bytes(packet_config[104:108], byteorder='little', signed=False) #int.from_bytes(packet_config[93:97], byteorder='little', signed=False)
    msgSize = int.from_bytes(packet_config[108:112], byteorder='little', signed=False) #int.from_bytes(packet_config[97:101], byteorder='little', signed=False)

    ping_config_bytes = {'masterMode'       : packet_config[0:1],
                         'pingRate_type'    : packet_config[1:2],
                         'user-gamma_correction': packet_config[2:3],
                         'usr_flags'        : packet_config[3:4],
                         'range_usr'        : packet_config[8:16],
                         'gainPercent_usr'  : packet_config[16:24],
                         'soundVel_usr'     : packet_config[24:32],
                         'salinity_usr'     : packet_config[32:40],
                         'pingId'           : packet_config[40:44],
                         'freq'             : packet_config[48:56],
                         'temp'             : packet_config[56:64],
                         'pressure'         : packet_config[64:72],
                         'soundVel_act'     : packet_config[72:80],
                         'pingStart'        : packet_config[80:84],
                         'dataSize_id'      : packet_config[84:85],
                         'rangeRes'         : packet_config[88:96],
                         'M'                : packet_config[96:98],
                         'N'                : packet_config[98:100],
                         'img_offset'       : packet_config[100:104],
                         'img_size'         : packet_config[104:108],
                         'msg_size'         : packet_config[108:112],
                         'leftovers'        : packet_config[112:] #packet_config[1:113]
                         }

    pingConfig =    {'master_mode'           : masterMode,
                     'pingRate_bytes'             : pingRate_type, #pingRate_dict[pingRate_type],
                     'user-gamma_correction' : gammaCorrection_usr,
                     'user-set_flags'        : usr_flags,
                     'user-stop_range'       : range_usr,
                     'user-gainPercent'      : gainPercent_usr,
                     'user-soundVel'         : soundVel_usr,
                     'user-salinity'         : salinity_usr,
                     'pingId'                : pingId,
                     'beam_frequency'        : freq,
                     'water_temperature'     : temp,
                     'water_pressure'        : pressure,
                     'soundVel_actual'       : soundVel_act,
                     'pingStart_time'        : pingStart,
                     'dataSize_bytes'        : dataSize_id + 1,
                     'range_resolution'      : rangeRes,
                     'N'                     : N,
                     'M'                     : M,
                     'brgTable_offset'       : 122,  # Magic number of the Bearing Table offset, based on how it's logged into the raw payload data & 
                                                     # information shared from DataStructure document. Need to be fixed in future versions, such that the 
                                                     # UWR logger also logs in the raw Bearing Table bytes separately, or also logs in the offset as extra info.
                     'image_offset'          : imgOffset,
                     'image_size'            : imgSize,
                     'message_size'          : msgSize
                    }
    if verbose:
        print(ping_config_bytes)
        print(pingConfig)
        print("length of config (bytes):", len(packet_config))
        #print(pingConfig)
    return pingConfig

def OculusPayload_SimpleFirePingResults_Extractor(payload_data, msg_version):
    """
    Helper function to properly parse SimpleFireMessage & SimplePingResults portions of payload_data,
    depending on the msg_version flag received in payload's OculusMessageHeader (V1 & V2 have slightly 
    different structures)

    Args:
    - payload_data => the Payload data's byte contents
    - msg_version => Number indicating version number of SimpleFireMessage & SimplePingResults 
    """
    masterMode = get_int_data(payload_data[16:17], False, byteorder='little')
    pingRate_type = payload_data[17:18]
    network_speed = payload_data[18:19]                                                        
    
    gammaCorrection_bytes = payload_data[19:20]
    if gammaCorrection_bytes == b'\x00':
        gammaCorrection_usr = 1.0
    else:
        gammaCorrection_usr = get_int_data(payload_data[19:20], False, byteorder='little')/255. * 1.0
    
    usr_flags       = payload_data[20:21]
    range_usr       = struct.unpack('d', payload_data[21:29])[0]
    gainPercent_usr = struct.unpack('d', payload_data[29:37])[0]
    soundVel_usr    = struct.unpack('d', payload_data[37:45])[0]
    salinity_usr    = struct.unpack('d', payload_data[45:53])[0]


    #V2
    print("version of message", msg_version)
    if msg_version == 2:
        ext_flags    = payload_data[53:57]
        print("Extended flags", ext_flags) #, struct.unpack('d', ext_flags)[0])
        reserved     = payload_data[57:89] #reserved consists of 8 * 4 bytes (uint32) characters 

        #parsing the OculusSimplePingResultV2 portion of the payload
        pingId       = int.from_bytes(payload_data[89:93], byteorder='little', signed=False)
        status       = payload_data[93:97]                                                           
        freq         = struct.unpack('d', payload_data[97:105])[0]
        temp         = struct.unpack('d', payload_data[105:113])[0]
        pressure     = struct.unpack('d', payload_data[113:121])[0]
        heading      = struct.unpack('d', payload_data[121:129])[0] #new info not available in V1
        pitch        = struct.unpack('d', payload_data[129:137])[0] #new info not available in V1
        roll         = struct.unpack('d', payload_data[137:145])[0] #new info not available in V1
        soundVel_act = struct.unpack('d', payload_data[145:153])[0]
        pingStart    = struct.unpack('d', payload_data[153:161])#[0] #pingStart using new data type
        dataSize_id  = int.from_bytes(payload_data[161:162], byteorder='little', signed=False)
        rangeRes     = struct.unpack('d', payload_data[162:170])[0]
        M            = int.from_bytes(payload_data[170:172], byteorder='little', signed=False)
        N            = int.from_bytes(payload_data[172:174], byteorder='little', signed=False)
        
        """#Note to self for where the spares of payload are located for V2
        spare0, spare1, spare2, spare3 = payload_data[174:178], payload_data[178:182], \
                                        payload_data[182:186], payload_data[186:190]
        """
        imgOffset    = int.from_bytes(payload_data[190:194], byteorder='little', signed=False)
        imgSize      = int.from_bytes(payload_data[194:198], byteorder='little', signed=False)
        msgSize      = int.from_bytes(payload_data[198:202], byteorder='little', signed=False)

        bearing_offset = 202

        ping_config_bytes = {'masterMode'       : payload_data[16:17],
                            'pingRate_type'    : payload_data[17:18],
                            'network_speed'    : payload_data[18:19],
                            'user-gamma_correction': payload_data[19:20],
                            'usr_flags'        : payload_data[20:21],
                            'range_usr'        : payload_data[21:29],
                            'gainPercent_usr'  : payload_data[29:37],
                            'soundVel_usr'     : payload_data[37:45],
                            'salinity_usr'     : payload_data[45:53],
                            'pingId'           : payload_data[89:93],
                            'status'           : payload_data[93:97],
                            'freq'             : payload_data[97:105],
                            'temp'             : payload_data[105:113],
                            'pressure'         : payload_data[113:121],
                            'heading'          : payload_data[121:129], #new information not available in V1
                            'pitch'            : payload_data[129:137], #new information not available in V1
                            'roll'             : payload_data[137:145], #new information not available in V1
                            'soundVel_act'     : payload_data[145:153],
                            'pingStart'        : payload_data[153:161],
                            'dataSize_id'      : payload_data[161:162],
                            'rangeRes'         : payload_data[162:170],
                            'M'                : payload_data[170:172],
                            'N'                : payload_data[172:174],
                            'img_offset'       : payload_data[190:194],
                            'img_size'         : payload_data[194:198],
                            'msg_size'         : payload_data[198:202]
                            }

    #V1
    else:
        pingId       = int.from_bytes(payload_data[53:57], byteorder='little', signed=False)
        status       = payload_data[57:61]                                                            # NOT USED /x/
        freq         = struct.unpack('d', payload_data[61:69])[0]
        temp         = struct.unpack('d', payload_data[69:77])[0]
        pressure     = struct.unpack('d', payload_data[77:85])[0]
        soundVel_act = struct.unpack('d', payload_data[85:93])[0]
        pingStart    = int.from_bytes(payload_data[93:97], byteorder='little', signed=False)
        dataSize_id  = int.from_bytes(payload_data[97:98], byteorder='little', signed=False)
        rangeRes     = struct.unpack('d', payload_data[98:106])[0]
        M            = int.from_bytes(payload_data[106:108], byteorder='little', signed=False)
        N            = int.from_bytes(payload_data[108:110], byteorder='little', signed=False)
        imgOffset    = int.from_bytes(payload_data[110:114], byteorder='little', signed=False)
        imgSize      = int.from_bytes(payload_data[114:118], byteorder='little', signed=False)
        msgSize      = int.from_bytes(payload_data[118:122], byteorder='little', signed=False)

        heading, pitch, roll = None, None, None
        bearing_offset = 122

        ping_config_bytes = {'masterMode'       : payload_data[16:17],
                            'pingRate_type'    : payload_data[17:18],
                            'network_speed'    : payload_data[18:19],
                            'user-gamma_correction': payload_data[19:20],
                            'usr_flags'        : payload_data[20:21],
                            'range_usr'        : payload_data[21:29],
                            'gainPercent_usr'  : payload_data[29:37],
                            'soundVel_usr'     : payload_data[37:45],
                            'salinity_usr'     : payload_data[45:53],
                            'pingId'           : payload_data[53:57],
                            'status'           : payload_data[57:61],
                            'freq'             : payload_data[61:69],
                            'temp'             : payload_data[69:77],
                            'pressure'         : payload_data[77:85],
                            'heading'          : None, 
                            'pitch'            : None, 
                            'roll'             : None, 
                            'soundVel_act'     : payload_data[85:93],
                            'pingStart'        : payload_data[93:97],
                            'dataSize_id'      : payload_data[97:98],
                            'rangeRes'         : payload_data[98:106],
                            'M'                : payload_data[106:108],
                            'N'                : payload_data[108:110],
                            'img_offset'       : payload_data[110:114],
                            'img_size'         : payload_data[114:118],
                            'msg_size'         : payload_data[118:122]
                            #'leftovers'        : payload_data[122:] #packet_config[1:113]
                            }

    pingConfig =    {'master_mode'           : masterMode,
                     'pingRate_bytes'        : pingRate_type, #pingRate_dict[pingRate_type],
                     'user-gamma_correction' : gammaCorrection_usr,
                     'user-set_flags'        : usr_flags,
                     'user-stop_range'       : range_usr,
                     'user-gainPercent'      : gainPercent_usr,
                     'user-soundVel'         : soundVel_usr,
                     'user-salinity'         : salinity_usr,
                     'pingId'                : pingId,
                     'beam_frequency'        : freq,
                     'water_temperature'     : temp,
                     'water_pressure'        : pressure,
                     'sensor_heading'        : heading,  #information not available in V1
                     'sensor_pitch'          : pitch,    #information not available in V1
                     'sensor_roll'           : roll,     #information not available in V1
                     'soundVel_actual'       : soundVel_act,
                     'pingStart_time'        : pingStart,
                     'dataSize_bytes'        : dataSize_id + 1,
                     'range_resolution'      : rangeRes,
                     'N'                     : N,
                     'M'                     : M,
                     'brgTable_offset'       : bearing_offset,  # Magic number of the Bearing Table offset, based on how it's logged into the raw payload data & 
                                                                # information shared from DataStructure document. 
                     'image_offset'          : imgOffset,
                     'image_size'            : imgSize,
                     'message_size'          : msgSize
                    }
                    
    return ping_config_bytes, pingConfig


def parse_packet_Oculusconfig(payload_data, verbose=False):
    '''
    [For use with '.oculus' files]
    Function to parse configuration information of a sonar ping object, 
    if using the default Oculus binary data file format (files with extension '.oculus')

    Note that this version extracts all configuration information that are stored within the payload data itself.
    '''

    pingRate_dict = {b'\x00': 10.00, b'\x01': 15.00, b'\x02': 40.00, b'\x03': 5.00, b'\x04': 2.00, b'\x05': None}

    #get payload's OculusMessageHeader data out first (no config data to be used in here - at least for now)
    oculusId = payload_data[0:2] #get_int_data(payload_data[0:2], False, byteorder='little')
    src_deviceId = payload_data[2:4]
    dst_deviceId = payload_data[4:6]
    message_Id = payload_data[6:8]
    message_version = get_int_data(payload_data[8:10], False, byteorder='big') #Indicator of either V1 vs V2 version of SimpleFireMessage & SimplePingResult
    firemessage_size = get_int_data(payload_data[10:14], False, byteorder='little')
    spare_bytes = payload_data[14:16]
    #oculus_head = payload_data[0:16]

    #SimpleFireMessage & SimplePingResult has V1 or V2 variants with slightly different structure.
    # Use message_version flag to properly parse config contents.

    #parsing the OculusSimpleFireMessage config information portion of payload 
    ping_config_bytes, pingConfig = OculusPayload_SimpleFirePingResults_Extractor(payload_data, message_version)
    
    
    if verbose:
        print(ping_config_bytes)
        print(pingConfig)
        print("Message version, OculusID bytes:", message_version, oculusId)
        print("Source, Destination Device ID:", src_deviceId, dst_deviceId)
        print("Message ID:", message_Id)
        print("length of Oculus Packet's core payload", pingConfig['message_size'])#msgSize)
    #sys.exit(0)
    return pingConfig


def init_OculusPing(payload_data, packet_size, payload_size, parsed_config, is_oculus_ext=True, verbose=False):
    '''
    Function to initialize and return a Sonar Ping object from the Blueprint Subsea Oculus sonars.
    '''
    
    #get the configuration data of the sonar ping.
    cur_config = None
    if is_oculus_ext:
        cur_config = parsed_config
    else:
        cur_config = parse_packet_UWRconfig(parsed_config, verbose=verbose)

    if verbose:
        print("Ping package size", packet_size)
        print("With the raw payload having size of: ", payload_size)
        print()

    #Create the Oculus FLS ping using the provided configuration
    Oculus_Ping = OculusFLSPing(pingConfig=parsed_config, verbose=verbose)

    #Add the initial raw data (?) to the created sonar ping object. 
    #init_water_col_data = payload_data

    #Update function returns flag indicating whether water column data is full (1) or not (0)!
    #if(sonar_ping.update_water_column_pixels(init_water_col_data)[0]) == 1:
    img_offset = cur_config['image_offset']
    brgTable_offset = cur_config['brgTable_offset']

    Oculus_Ping.update_acousticImg_raw(payload_data, img_offset=img_offset, verbose=verbose)
    Oculus_Ping.update_brgTable_raw(payload_data, brgTable_offset=brgTable_offset, verbose=verbose)
    #if(sonar_ping.update_water_column_bytes(init_water_col_data)[0]) == 1:
    #    return None
    return Oculus_Ping

def OculusPing_generator(path, n=None, start_ping=0, no_ping_limit=100, verbose=False):
    '''
    Generator function to extract n number of Ping Objects from a custom UWR Oculus log file.

    Arguments:
    - path =>
    - n => (Optional)
    - start_ping => (Optional)
    - no_ping_limit => (Optional)
    - verbose =>

    Returns:
    None. Generator *yields* (n - start_ping) amount of ping objects if called to a list object 
    '''
    
    ping_counter = 0
    byte_counter = 0
    no_ping_counter = 0
    is_oculus_ext = path.split(".")[-1] == "oculus" #check whether file is using the '.oculus' extension or not
    if verbose:
        print(path.split("."))


    with open(path, "rb") as bin_file:
        fileHeader = bin_file.read(4)
        fileHeader_size = bin_file.read(4)
        source = bin_file.read(16)
        version = bin_file.read(2)
        print("Is oculus file?", is_oculus_ext)

        if is_oculus_ext: 
            encryption = bin_file.read(2)
            key = bin_file.read(8)
            buffer = bin_file.read(4) #weird buffer still exists for .oculus files

        else:
            buffer = bin_file.read(6) #For an unknown reason, it seems that the Qt IO Writing introduced a weird '\x00'(hex form) bytes each time
                                  #a double data type is written. (look at line 311 in the parse_PingPacket function for other case)
        file_time = bin_file.read(8)
        if verbose:
            print("path file created on:", file_time)

        #Note that originally in the Qt application, the header was supposed to be written as 0x11223344 (Hex format). Reading in the binary file
        #in Python right now would yield b'\x44223311' instead, which is why the checking is using the reversed Hex format instead. 
        if fileHeader.hex() != '44332211': 
            print(fileHeader)
            print("Wrong file header format received! Please ensure the file type is correct!")
            sys.exit(0)

        packet_range = None
        if n is not None:
            packet_range = range(0, n)
        else:
            packet_range = itertools.count()

        for i in packet_range:
            byte_counter = 0
            packet_counter = 0

            timestamp, packet_size, payload_size, packet_config, payload_data, ping_valid = parse_PingPacket(bin_file, is_oculus_ext, verbose=verbose)
            byte_counter += packet_size
            packet_counter += 1
            if ping_valid:
                print("Found valid ping object. Collecting data for ping #:", ping_counter)
                if not is_oculus_ext:
                    parsed_config = parse_packet_UWRconfig(packet_config, verbose=verbose)
                else:
                    parsed_config = parse_packet_Oculusconfig(payload_data, verbose=verbose)

                parsed_config["pingStart_time"] = timestamp
                sonar_ping = init_OculusPing(payload_data, packet_size, payload_size, parsed_config, is_oculus_ext, verbose=verbose)
                if verbose:
                    print("byte-packet counter:", byte_counter, packet_counter)
                if sonar_ping == None:
                    print("Failed to create sonar ping object!")
                    sys.exit(1)
                elif i >= start_ping:
                    yield sonar_ping
            else:
                no_ping_counter += 1
                if no_ping_counter == no_ping_limit:
                    break
            '''
            if i >= start_ping and ping_valid is not False:
                yield sonar_ping #collected_sonar_pings.append(sonar_ping)
            '''
            print("Done extracting data for ping #", ping_counter)
            print()
            ping_counter += 1
            
    bin_file.close()

    print("File is confirmed to be a valid Oculus file, with source from: ", source, ". Version:", version)

    print()
    return

if __name__ == "__main__":

    collected_sonar_pings = []
    ping_counter = 0
    prev_is_ping_header = False
    prev_packet_data = None 
    '''
    with open(binfile_path, "rb") as bin_file:
        fileHeader = bin_file.read(4)
        fileHeader_size = bin_file.read(4)
        source = bin_file.read(16)
        version = bin_file.read(2)
        buffer = bin_file.read(6) #For an unknown reason, it seems that the Qt IO Writing introduced a weird '\x00'(hex form) bytes each time
                                  #a double data type is written. (look at line 505 for other case)
        file_time = bin_file.read(8)
        
        #Note that originally in the Qt application, the header was supposed to be written as 0x11223344 (Hex format). Reading in the binary file
        #in Python right now would yield b'\x44223311' instead, which is why the checking is using the reversed Hex format instead. 
        if fileHeader.hex() != '44332211': 
            print(fileHeader)
            print("Wrong file header format received! Please ensure the file type is correct!")
            sys.exit(0)

        for i in range(0, 280):
            byte_counter = 0
            packet_counter = 0
            verbose = False
            #Case where last packet from previous loop iteration is actually start of a new ping.
            if prev_packet_data != None and prev_is_ping_header:
                sonar_ping = init_OculusPing(prev_packet_data, packet_size, payload_size, packet_config, verbose=True)
                if sonar_ping == None:
                    print("Failed to create sonar ping object!")
                    sys.exit(1)
            else:
                timestamp, packet_size, payload_size, packet_config, payload_data, valid = parse_PingPacket(bin_file, verbose=True)
                byte_counter += packet_size
                packet_counter += 1
                if valid:
                    print("Found valid ping object. Collecting data for ping #:", ping_counter)
                    parsed_config = parse_packet_UWRconfig(packet_config, verbose=True)
                    sonar_ping = init_OculusPing(payload_data, packet_size, payload_size, packet_config, parsed_config=parsed_config, verbose=True)
                    if verbose:
                        print("byte-packet counter:", byte_counter, packet_counter)
                    if sonar_ping == None:
                        print("Failed to create sonar ping object!")
                        sys.exit(1)
            
            #prev_is_ping_header, prev_packet_data = ping_capture_helper(sonar_ping, bin_file, byte_counter, packet_counter, verbose=True)
            prev_is_ping_header = False

            collected_sonar_pings.append(sonar_ping)
            print("Done recording data for ping #", ping_counter)
            print()
            ping_counter += 1
            
    bin_file.close()
    print()
    '''
    Oculus_generator = OculusPing_generator(binfile_path, n=10, verbose=True)
    collected_sonar_pings = [oculus_ping for oculus_ping in Oculus_generator]

    print("Collected Sonar ping samples:")
    start = 0
    for i, sonar_ping in enumerate(collected_sonar_pings[start : start+10]):
        
        config = sonar_ping.pingConfig
        #watercol_matrix = sonar_ping.water_column_matrix
        #beamdirs = sonar_ping.beam_dirs
        print("Configs for sonar ping sample #", start + i+1)
        print(config)
        #print("Water column data Full?", sonar_ping.is_pixels_filled())
        #print("Beam dirs data Full?", sonar_ping.is_beamdirs_filled())
        #print("Shape of water column:", watercol_matrix.shape)
        #print("Shape of beam directions:", beamdirs.shape)
        #print(watercol_matrix.max(), watercol_matrix.min())
        print(vars(sonar_ping)) # uncomment to see full details of the ping data
        print()


