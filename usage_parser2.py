import glob
from datetime import datetime, timedelta
import re 

import matplotlib 
matplotlib.use('Agg')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from itertools import islice,chain
import collections

import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages


MEM_LIST=['max_rss_GB', 'max_vms_GB']
IO_LIST=[('total_io_read_GB'), ('total_io_write_GB')]


class Log_Data():
	def __init__ (self, **kwargs): 
		self._parent_folder = None 		#Parent folder name -- contains all the folders with log files 
		self._image_size = 62			#Image size -- max number of jobs to be plotted

		for varname, value in kwargs.items():	#Creates a class attribute for each job type specified
			setattr(self, varname, value)

	def get_attr(self,name):
		""" Returns the class attribute named as the given name """
		return object.__getattribute__(self, name)

	def modify_attr(self,name,val): 
		""" Sets the value (val) for a given attribute (name) """
		object.__setattr__(self,name,val)

	def set_image_size(self, sz):
		self._image_size = sz


	def set_parent_folder(self, pth): 
		self._parent_folder = pth

	@property
	def image_size(self):
		return self._image_size
	

	@property
	def parent_folder(self):
		return self._parent_folder


conversion_dict = {'K': -2, 'M': -1, 'G': 0}

def conversion(x):
    """ Converts a given number (x) into a more readable format """
    return x[-1] in conversion_dict and float(x[:-1])*1024.**conversion_dict[x[-1]] or 0.0

def compute_df_columns(df):
	if len(df['timestamp']) == 0:
		return None

	df['timestamp'] = df['timestamp'].apply(datetime.strptime, args=('%Y-%m-%dT%H:%M:%S.%f',))
	df['time_delta_s'] = (df['timestamp'] - df['timestamp'].shift(1)).apply(lambda x: x.total_seconds())
	df['time_spent'] = df['timestamp'] - df['timestamp'][0]
	df['time_spent_s'] = df['time_spent'].apply(lambda x: x.total_seconds())
	total_duration = df['time_spent_s'].iloc[-1]


	if np.isclose(total_duration, 0):
		df['time_spent_rel'] = 0.
	else:
		df['time_spent_rel'] = df['time_spent_s']/df['time_spent_s'].iloc[-1]

	if 'max_vms' in df.columns:
		df['max_vms_GB'] = df['max_vms'].apply(conversion)
	if 'max_rss' in df.columns:
		df['max_rss_GB'] = df['max_rss'].apply(conversion)
	if 'max_uss' in df.columns:
		df['max_uss_GB'] = df['max_uss'].apply(conversion)
	if 'total_io_read' in df.columns:
		df['total_io_read_GB'] = df['total_io_read'].apply(conversion)
	if 'total_io_write' in df.columns:
		df['total_io_write_GB'] = df['total_io_write'].apply(conversion)
	if 'total_cpu_time' in df.columns:
		df['cpu_perc'] = 100.*(df['total_cpu_time'] - df['total_cpu_time'].shift(1))/df['time_delta_s']

	return df

class Usage_Parser2(): 
	def __init__(self, lgdr, jobs, mem=4, wr=10 ): 
		self.log_path = lgdr		#All log files from a folder
		self.data = []				#Contains the data of all the jobs separated by parent folder (list:Log_Data())
		self.selected_jobs = jobs 	#job types that will be used to plot/get stats 

		self.memory_limit = mem   	
		self.iow_limit = wr

	def load_usage_files(self): 
		""" Loads all the usage files for the selected jobs into self.data using Log_Data class. """
		folders = glob.glob(self.log_path)

		for fld in folders: 
			usg = glob.glob(fld+"/*/usage_*")
			dic = {}

			for j in self.selected_jobs:
				p = re.compile('(.*)'+j)
				
				for u in usg:
					if re.search(p, u):
						if j in dic:
							dic[j].append(u)
						else : 
							dic[j] = [u]
			if dic :
				ld = Log_Data(**dic)
				ld.set_parent_folder(fld)

				self.data.append(ld)
		

	def order_usages(self, log_files, j, set_size=False):
		""" Loads log_files data by job type and orders them according to their identifier
	
			Parameters
			----------
				log_files: 	list 
							List of log files (usage_*)

				j:	str
					Name of the job type to get the data from 

				set_size: 	bool
							Used to set image_size

			Return
			------
				ordered_data: 	collections.OrderedDict()
								Ordered dictionary where keys are the orderd identifier of the jobs and values are its assigned log data 
				size: 	int 
					 	set_size=True: returns the number of jobs of the given type (j) 
					 	set_size=False: returns 0 
		"""
		keys = []
		to_be_ordered = {}

		for usg in log_files:
			df = pd.read_csv(usg, engine='python')
			compute_df_columns(df)

			if not df.empty: 
				nums = re.findall(r'\d+', usg.split('/')[-2])
				n = ''
				for i in range(len(nums)): 
					n += str(nums[i])+'.'
				n = n[:-1]	#le quitamos el ultimo punto 

				to_be_ordered[n] = df
				keys.append(n)

		keys.sort(key=lambda s: list(map(int, s.split('.'))))
		ordered_data = collections.OrderedDict()

		for k in keys: 
			ordered_data[k] = to_be_ordered[k]

		size=0
		if set_size : 
			match = '1.1.'
			for k in keys : 
				if match in k:
					size +=1

		return ordered_data,size


	def load_data(self, jobs_to_order, set_size_job=False):
		"""	Loads all the logs data into self.data. 

			Parameters
			----------
				jobs_to_order: 	list
								Types of jobs that will need to be ordered by id	

				set_size_job;	str
								Job that will be used to fix the size of the plots 
		"""
		self.load_usage_files()		#Loads ALL the usage files from the parent folder 


		i=0
		for d in self.data:
			i +=1
			for j in self.selected_jobs: 
				if j in jobs_to_order:
					usg = d.get_attr(j)

					if len(usg) > 1 :
						
						dfs = {}
						if set_size_job == j : 
							size = 0
							dfs,size = self.order_usages(usg,j, True)
							
							d.set_image_size(size)
						else : 
							dfs,size = self.order_usages(usg,j)
					else :
						dfs = []
						#print (j, ' -- ', usg) 
						df = pd.read_csv(usg.pop(), engine='python')
						compute_df_columns(df)

						if not df.empty:
							dfs.append(df)
				else : 
					for usg in d.get_attr(j): 
						dfs = []

						df = pd.read_csv(usg, engine='python')
						compute_df_columns(df)

						if not df.empty: 
							dfs.append(df)

				d.modify_attr(j,dfs)

	def split_data(self, d, max_size=62): 
		""" Creates a list containing the indexes of data splitted in chunks  

			Parameters
			----------
				d: data  
				max_size: int 
					Maximum size of the data to be sh

			Return
			------
				chunk: 	list
						List containing a list for each plot that will be made.

		"""
		chunk = []
		sample_size = 0
		add = d.image_size
		x = 0

		for j in self.selected_jobs: 
			sample_size += len(d.get_attr(j))
		#print (j, ' -- sample size : ', sample_size)
		if max_size < d.image_size:
			add = max_size

		#print (d.parent_folder ," -- ",  d.image_size) 

		while x+add < sample_size :
			chunk.append([x,x+add-1]) 
			x+=add 

		if x != sample_size: 
			chunk.append([x,sample_size]) 

		return chunk 


	def key_and_value(self, data):
		""" Gets the keys and data from the given Log_Data """

		data_list = []
		key_list = []

		for j in self.selected_jobs: 
			if type(data.get_attr(j)) is collections.OrderedDict :
				data_list = list(chain(data_list, data.get_attr(j).values()))
				key_list = list(chain(key_list, (data.get_attr(j).keys())))
			else : 
				
				
				for d in data.get_attr(j):
					data_list.append(d)
					
					#print (d, j)
				key_list.append(j)

		return data_list, key_list

	def adjust_spines(self, ax,spines):
		""" Used to remove the an axis from a plot """

		for loc, spine in ax.spines.items():
			if loc in spines:
				spine.set_position(('outward',10)) # outward by 10 points
			else:
				spine.set_color('none') # don't draw spine

		# turn off ticks where there is no spine
		if 'left' in spines:
			ax.yaxis.set_ticks_position('left')
		else:
		# no yaxis ticks
			ax.yaxis.set_ticks([])

		if 'bottom' in spines:
			ax.xaxis.set_ticks_position('bottom')
		else:
		# no xaxis ticks
			ax.xaxis.set_ticks([])


	def io_plot(self, fig, gs, data, chunk, io_write_limit=None,var_list=IO_LIST): 
		""" Plots the IO reads and writes, if the data to be plotted (write) passes io_write_limit it will be shown 	
			in red, if not it'll be green. 

			Parameters
			----------
				fig: Figure where the plot will be saved

				gs: matplotlib.gridspec.GridSpec()
					Set the position of the plot inside the figure (fig)

				data: 	Log_Data
						Data that has to be plotted 

				chunk: 	list 
						Indexes of the data to be plotted (chunk[0] = index of the beginning, chunk[-1]= index of the end)
	
				io_write_limit: 	int 
									IOw threshold that shouldn't be surpassed by a job
	
				var_list: 	list
							IOw/r variables to be plotted, defined at the beginning of the script. 


		"""

		ax_ind = 0
		chunk_size = chunk[-1]-chunk[0]

		if io_write_limit == None : 
			io_write_limit = self.iow_limit

		for var in var_list:
			ax = plt.subplot(gs[2,ax_ind])
			jobs_data, jobs_key = self.key_and_value(data)


			var_max = []
			for v in islice(jobs_data, chunk[0], chunk[1]+1):
				var_max.append(max(v[var]))

			#FALTA PLANNER
			ax_ind +=1 
			plt.xticks(np.arange(0,chunk_size+1), list(jobs_key)[chunk[0]:chunk[-1]+1], rotation='vertical')

			plt.tick_params(axis='x', labelsize=6)

			plt.plot(np.arange(0, len(var_max)),var_max,'o',label=var, markersize=2.4)
			ax.grid(linestyle='dashdot')
			
			plt.title(str(var))
		fig.align_labels()


	def memory_plot(self, fig, gs, data, chunk, mem_limit=None, var_list=MEM_LIST ):
		""" Plots the memory (vms & rss), if rss > mem_limit -> job data will be printed red on the plot, otherwise it will be green. 

			Parameters
			----------
				fig: Figure where the plot will be saved

				gs: matplotlib.gridspec.GridSpec()
					Set the position of the plot inside the figure (fig)

				data: 	Log_Data
						Data that has to be plotted 

				chunk: 	list 
						Indexes of the data to be plotted (chunk[0] = index of the beginning, chunk[-1]= index of the end)

				mem_limit: 	int 
							memory threshold that shouldn't be surpassed by a job

				var_list: 	list
							Memory (rss/vms) variables to be plotted, defined at the beginning of the script. 

		"""

		ax_ind = 0
		i=0

		if mem_limit == None : 
			mem_limit = self.memory_limit

		chunk_size = chunk[-1]-chunk[0]

		for var in var_list:
			ax = plt.subplot(gs[ax_ind,:])	
			ax.grid(linestyle='dashdot')	

			jobs_data, jobs_key = self.key_and_value(data)

			mn = {}
			serror = []

			for v in islice(jobs_data, chunk[0], chunk[1]+1): 
				#print (type(v))
				mn[i]=np.mean(v[var])
				serror.append(np.std(v[var]))

				bp = ax.scatter(np.zeros_like(v[var]) + i,v[var], s=8)


				if var == 'max_rss_GB' :
					if max(v[var]) > mem_limit : 
						bp.set_facecolor('tomato')
					else:
						bp.set_facecolor('yellowgreen') 
				else : 
					bp.set_facecolor('yellowgreen') 

				i+=1

			if ax_ind == 0 : 
				plt.margins(y=0.6)
				self.adjust_spines(ax, ['left'])
			else : 
				self.adjust_spines(ax, ['left','bottom'])


			plt.ylim(ymin=0)
			plt.title(str(var))
			ax.errorbar(mn.keys(),mn.values(),  yerr=serror, color='black', elinewidth=1, alpha=0.7)
			ax_ind +=1
			
		plt.xticks(np.arange(chunk_size+1,chunk_size*2+2), list(jobs_key)[chunk[0]:chunk[-1]+1], rotation='vertical')


	def cpu_perc_plot(self, fig, gs, data, chunk=None): 
		""" Plots the usage in % of the cpu by the jobs on data. Prints the usage on different colors, usg<25% = red , 25%<usg<50% = 
		yellow, 50%<usg<75% dark green, 75%<usg neon green.

			Parameters
			----------
				fig: Figure where the plot will be saved

				gs: matplotlib.gridspec.GridSpec()
					Set the position of the plot inside the figure (fig)

				data: 	Log_Data
						Data that has to be plotted 

				chunk: 	list 
						Indexes of the data to be plotted (chunk[0] = index of the beginning, chunk[-1]= index of the end)

		"""

		ax = plt.subplot(gs[3,:])	
		mn = {}
		chunk_size = chunk[-1]-chunk[0]
		i=0

		jobs_data, jobs_key = self.key_and_value(data)

		for v in islice(jobs_data, chunk[0], chunk[1]+1):
			mn[i]= np.mean(v['cpu_perc'])
			perc_mn = np.mean(v['cpu_perc'])
			plt.title('cpu_perc')

			plt.xticks(np.arange(0,chunk_size*2+2), jobs_key[chunk[0]:chunk[-1]+1], rotation='vertical')

			bp = ax.scatter(np.zeros_like(v['cpu_perc']) + i,v['cpu_perc'], s=3)
			if perc_mn < 25 : 
					bp.set_facecolor('tomato')
			elif perc_mn > 25 and perc_mn < 50:
				bp.set_facecolor('yellow') 
			elif perc_mn > 50 and perc_mn < 75 :
				bp.set_facecolor('mediumseagreen') 
			else:
				bp.set_facecolor('lime') 
			
			ax.grid(linestyle='dashdot')
			#ax.errorbar(perc_mn.keys(),perc_mn.values(),  color='black', elinewidth=1, alpha=0.7)
			i+=1
		ax.plot(list(mn.keys()),list(mn.values()),color='black', linewidth=1, alpha=0.7)
	

	def get_mean_and_max(self,job_name,data):
		""" Computes max and mean value for a type of job, prints the results 
			
			Parameters
			----------
			job_name : str
				String that contains the name of type of job to analize 

			data : dictionary 
				All the data for a folder of the job to be analized
				key : 	job identifier 
				value : data values  
		"""

		max_time = 0
		max_ram = 0
		max_IOW = 0
		max_cpuperc = 0

		mt = 0	#mean_time
		mr = 0	#mean_rss
		mwr = 0	#mean_ioWrite
		mcpu = 0


		if not isinstance(data, list):
			data_f = data.values()
			
		else : 
			data_f = data

		for d in data_f : 			
			mr += np.mean(d['max_rss_GB'])
			mcpu += np.mean(d['cpu_perc'])

			tmp = np.array(d['time_spent_s'])[-1]
			if max_time < np.array(d['time_spent_s'])[-1] : 
				max_time = tmp 
			mt += tmp

			tmp = np.max(d['max_rss_GB'])
			if max_ram < tmp: 
				max_ram = tmp 

			tmp = np.max(d['cpu_perc'])
			if max_cpuperc < tmp: 
				max_cpuperc = tmp 

			tmp = np.array(d['total_io_write_GB'])[-1]
			if max_IOW < tmp : 
				max_IOW = tmp 
			mwr += tmp

		
		mean_time = mt/len(data)
		mean_ram = mr/len(data)
		mean_IOW = mwr/len(data)
		mean_cpuperc = mcpu/len(data)

		mean_time = int(mean_time/60)+(mean_time%60)*0.01
		max_time = int(max_time/60)+(max_time%60)*0.01

		print (job_name, " MEAN TIME (min.secs) : %.2f " % (mean_time), " -- MAX TIME (min.secs) : %.2f"  % max_time)
		print (job_name," MEAN CPU_PERCENT : %.2f" % mean_cpuperc, " -- MAX CPU_PERCENT : %.2f " % max_cpuperc)
		print (job_name," MEAN MEMORY (GB) : %.2f" % mean_ram, " -- MAX MEMORY : %.2f"  % max_ram)
		print (job_name," MEAN IOW (GB) : %.2f" % mean_IOW, " -- MAX IO/W : %.2f"  % max_IOW)

	def get_job_stats(self): 
		"""	Gets a summary of the max / mean for the given data
			
			Parameters 
			----------
			job_type: list 
				All the job types on the folder ([splitter, TU, detector]) 

			data:	
				key : parent folder where stats will be computed. 
				value : dict() || list 
		"""
		print ("JOB STATS : ")
		#key == folder_name | val == cvs info from key folder  
		for val in self.data: 
			#Prints MAX/PEAK values for cpu%, time(s), rss(GB) and GB written by all jobs of the same kind  
			print('')
			print ("CARPETA : ", val.parent_folder, " ______ ")
			print('')

			for j in self.selected_jobs :
				print ('--- '+j+' --- ')
				print ('-------------------- ')
				self.get_mean_and_max(j,val.get_attr(j))
				print('')

			print ('___________________________')
		print ('')


	def plot_all_jobs(self, max_size=62):
		""" Plots all the jobs on the folders on self

			Jobs will be classified per folder and plotted on different pdfs, every page of the pdf will contain 2 memory plots (rss/vms)
			2 IO plots (read/write) and a cpu usage (%) plot. 


		"""
		
		for d in self.data: 
			with PdfPages(('/nfs/pic.es/user/c/cperalta/python_envs/python_3.5.1/cosasAcabadas/parser/plots/'+d.parent_folder.split('/')[-2]+'.pdf')) as pdf:
				print ("CARPETA : ", d.parent_folder, " ______ ")
				cs = self.split_data(d, max_size)

				for c in cs : 
					fig = plt.figure(figsize=(10, 15))	#,tight_layout=True
					gs = gridspec.GridSpec(4, 2, hspace=0.05, top=0.94)
					plt.suptitle("--" +str(d.parent_folder), fontsize=12)
					self.memory_plot(fig, gs, d,c)
					gs2 = gridspec.GridSpec(4, 2, hspace=0.3)
					self.io_plot(fig,gs2,d,c)
					self.cpu_perc_plot(fig,gs2,d,c)
					pdf.savefig()
				print ("done")
