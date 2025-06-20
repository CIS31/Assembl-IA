import pandas as pd
import psycopg2

class AzureUtils:
    def __init__(self, mount_dir):
        self.mount_dir = mount_dir

    def detect_azure_run(self):
        """
        Function to detect if the code is running in an Azure environment.
        """
        args = dict(arg.split('=') for arg in sys.argv[1:] if '=' in arg)
        return args.get("AZURE_RUN", "false").lower() == "true"

    def mount_dir_Azure(self):
        """
        Function to mount the directory in Azure environment.
        """
        def is_mounted(mount_point):
            mounts = [mount.mountPoint for mount in dbutils.fs.mounts()]
            return mount_point in mounts

        configs = {
            "fs.azure.account.auth.type": "OAuth",
            "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
            "fs.azure.account.oauth2.client.id": dbutils.secrets.get(scope="az-kv-assemblia-scope", key="sp-application-id"),
            "fs.azure.account.oauth2.client.secret": dbutils.secrets.get(scope="az-kv-assemblia-scope", key="sp-secret-value"),
            "fs.azure.account.oauth2.client.endpoint": f"https://login.microsoftonline.com/{dbutils.secrets.get(scope='az-kv-assemblia-scope', key='sp-tenant-id')}/oauth2/token"
        }

        if not is_mounted(self.mount_dir):
            dbutils.fs.mount(
                source="abfss://data@azbstelecomparis.dfs.core.windows.net/",
                mount_point=self.mount_dir,
                extra_configs=configs
            )
            print(f"Successfully mounted {self.mount_dir}")
        else:
            print(f"{self.mount_dir} is already mounted")

    def get_latest_wav(self, blob_folder):
        """
        Function to get the latest .wav file (by modification date) in a blob folder.
        """
        files = dbutils.fs.ls(blob_folder)
        wav_files = [f for f in files if f.name.endswith('.wav')]
        if not wav_files:
            raise FileNotFoundError(f"No .wav files found in {blob_folder}")

        latest_file = sorted(wav_files, key=lambda f: f.modificationTime, reverse=True)[0]
        print(f"Latest .wav file: {latest_file.path} (modified at {latest_file.modificationTime})")
        return latest_file.path
    
    def get_latest_xml(self, blob_folder):
        """
        Function to get the latest .xml file (by modification date) in a blob folder.
        """
        files = dbutils.fs.ls(blob_folder)
        xml_files = [f for f in files if f.name.endswith('.xml')]
        if not xml_files:
            raise FileNotFoundError(f"No .xml files found in {blob_folder}")

        latest_file = sorted(xml_files, key=lambda f: f.modificationTime, reverse=True)[0]
        print(f"Latest .xml file: {latest_file.path} (modified at {latest_file.modificationTime})")
        return latest_file.path
    

class PostgresUtils:
    def __init__(self):
        """
        Initialize the PostgreSQL connection using environment variables.
        """
        # Parse the arguments passed to the script as "key=value"
        args = dict(arg.split('=') for arg in sys.argv[1:] if '=' in arg)

        # Inject them as environment variables.
        os.environ['PGHOST'] = args.get('PGHOST', '')
        os.environ['PGDATABASE'] = args.get('PGDATABASE', '')
        os.environ['PGUSER'] = args.get('PGUSER', '')
        os.environ['PGPASSWORD'] = args.get('PGPASSWORD', '')
        os.environ['PGPORT']     = args.get('PGPORT', os.getenv('PGPORT', '5432'))

        self.host = os.getenv('PGHOST')
        self.database = os.getenv('PGDATABASE')
        self.user = os.getenv('PGUSER')
        self.password = os.getenv('PGPASSWORD')
        self.port = int(os.getenv('PGPORT', 5432))
        self.conn = None

        print(f"PostgreSQL connection parameters: host={self.host}, database={self.database}, user={self.user}, port={self.port}")

    def connect(self):
        """
        Establish a connection to the PostgreSQL database.
        """
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port,
                sslmode="require"
            )
            print(f"Connected to PostgreSQL database: {self.database}")
        except Exception as e:
            print(f"Error connecting to PostgreSQL: {e}")
            raise

    def create_export_table(self, table_name):
        """
        Create a table for storing exported data if it doesn't exist.
        :param table_name: Name of the table to create.
        """
        try:
            cursor = self.conn.cursor()
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                filename TEXT,
                speaker_label TEXT,
                start DOUBLE PRECISION,
                duration DOUBLE PRECISION,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            cursor.execute(create_table_query)
            self.conn.commit()
            cursor.close()
            print(f"Table '{table_name}' for exported data created successfully.")
        except Exception as e:
            print(f"Error creating export table '{table_name}': {e}")
            raise

    def insert_csv_to_table(self, table_name, csv_file_path, columns_map):
        """
        Insert data from a CSV file into the specified PostgreSQL table.
        :param table_name: Name of the table to insert data into.
        :param csv_file_path: Path to the CSV file.
        :param columns_map: Ordered list of column names in the table matching the CSV columns.
        """

        try:
            df = pd.read_csv(csv_file_path)
            cursor = self.conn.cursor()
            insert_query = f"""
                INSERT INTO {table_name} ({', '.join(columns_map)})
                VALUES ({', '.join(['%s'] * len(columns_map))});
            """
            for _, row in df.iterrows():
                values = [row[col] for col in columns_map]
                cursor.execute(insert_query, values)
            self.conn.commit()
            cursor.close()
            print(f"CSV data from '{csv_file_path}' successfully inserted into table '{table_name}'.")
        except Exception as e:
            print(f"Error inserting CSV data into table '{table_name}': {e}")
            raise

    def close_connection(self):
        """
        Close the connection to the PostgreSQL database.
        """
        if self.conn:
            self.conn.close()
            print("PostgreSQL connection closed.")
