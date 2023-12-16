from urllib import request
from bs4 import BeautifulSoup
import os
import fnmatch
import PyPDF2 
from alive_progress import alive_bar
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn. feature_extraction.text import TfidfVectorizer
import pandas as pd
import circlify
import seaborn as sns
import matplotlib.pyplot as plt

class BooksAnalysis:
    def __init__(self, url, pdf_dir, txt_dir, png_dir):
        self._url = url
        self._pdf_dir = pdf_dir
        self._txt_dir = txt_dir
        self._png_dir = png_dir

    def get_urls(self):
        """Obtener las URL de los libros del sitio web del repositorio."""
        page = request.urlopen(self._url)
        soup = BeautifulSoup(page, 'html.parser')

        try:
            divs = soup.find_all('div', class_='bookContainer grow')
            with open('book_urls.txt', 'w') as fd:
                for div in divs:
                    book_url = div.findChild('a', href=True)['href']
                    download_url = self._url + book_url
                    download_page = request.urlopen(download_url)
                    download_soup = BeautifulSoup(download_page, 'html.parser')
                    footer = download_soup.find('div', {'id': 'footer'})
                    file_name = footer.contents[0]
                    full_url = download_url + file_name
                    fd.write(full_url + '\n')
        except:
            pass

    def get_pdfs(self):
        """Running the desktop automation Shell Script
        """
        os.system("./get_books.sh")

    def to_text(self):
        """Extracting the PDF text.
        """
        total = len(fnmatch.filter(os.listdir(self._pdf_dir), '*.pdf'))
        counter = 1
        print(PyPDF2.__version__)
        for filename in os.listdir(self._pdf_dir):
            pdfFileObj = open(os.path.join(self._pdf_dir, filename), 'rb')
            try:
                pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
                text_filename = filename.replace('.pdf', '.txt') 
                with open(os.path.join(self._txt_dir, text_filename), "w") as f:
                    print(f'Book {counter}/{total}: {filename}...')
                    with alive_bar(pdfReader.numPages) as bar:
                        for page in range(pdfReader.numPages):
                            pageObj = pdfReader.getPage(page)
                            f.write(pageObj.extractText() + '\n\n')
                            bar()
                    pdfFileObj.close()
                counter += 1
            except Exception as e:
                print(f"Error al procesar el archivo {filename}: {str(e)}")
                pass
    
    def word_freq(self):
        """Computing the word frequencies.
        """
        for filename in os.listdir(self._txt_dir):
            fullname = os.path.join(self._txt_dir, filename)
            if os.path.isfile(fullname):
                print(f'Processing {filename}...')
                relevant = []
                with open(fullname, 'r') as fd:
                    try:
                        doc = fd.readlines()
                        relevant = self.get_relevant_words(30, doc)
                        self.word_cloud(relevant, filename)
                    except Exception as e:
                        print(f"Error al procesar el archivo {filename}: {str(e)}")
                        pass

    def get_relevant_words (self, word_count, doc): 
        """Cleans the list of non-alphanumeric characters
        """
        data = []
        total = len(doc)
        with alive_bar(total) as bar:
            for statement in doc:
                clean_data = re.sub('[^a-zA-Z]', ' ', statement)
                tokens = word_tokenize(clean_data)
                tokens = [word.lower() for word in tokens]
                if len(tokens):
                    statement = ''
                    for word in tokens:
                        if not word in stopwords.words():
                            statement += word. lower() + ''
                        if len(statement.strip()) > 0:
                            data.append(statement.strip())
                bar()
        tr_idf_model = TfidfVectorizer (ngram_range=(1,1))
        tf_idf_vector = tr_idf_model.fit_transform(data)
        weights = [(word, tf_idf_vector.getcol(idx).sum()) 
                    for word, idx in tr_idf_model.vocabulary_.items()]
        weights.sort(key=lambda i:i[1], reverse=True)
        if len(weights) < word_count:
            return weights
        else:
            return weights[:word_count]
        
    def word_cloud(self, data, filename):
        """Building of the word-cloud."""
        df_words =pd.DataFrame(data=data, columns=['words', 'count'])
        circles = circlify.circlify(
            df_words['count'].tolist(),
            show_enclosure=False,
            target_enclosure=circlify.Circle(x=0, y=0)
        )
        n = df_words['count'].max()
        color_dict = self.get_color_dict(sns.color_palette("pastel"), n,1)

        fig, ax = plt.subplots(figsize=(9,9), facecolor='white')
        ax.axis('off')
        lim = max(max(abs(circle.x) + circle.r, abs(circle.y) + circle.r,) for circle in circles)
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)

        labels = list(df_words['words'])
        counts = list(df_words['count'])
        labels.reverse()
        counts.reverse()

        #print circles
        for circle, label, color in zip(circles, labels, counts):
            x, y, r = circle
            ax.add_patch(
                plt.Circle(
                    (x, y),
                    r, 
                    alpha=0.9,
                    linewidth=2,
                    color=color_dict.get(int(color * 100))
                )
            )
            plt.annotate(label, (x, y), size=12, va='center', ha='center')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(self._png_dir, filename.replace('.txt', '.png')))
        
    def get_color_dict(self, palette, number, start):
        """Creating of the color-palette.
        
        Args:
            palette (string): Color-palette name.
            number (int): Number of colors.
            start (int): Initial color_code.

        Returns:
        dictionary: Colors dictionary.
        """
        number = int(number *100)
        pal = list(sns.color_palette(palette=palette, n_colors=number).as_hex())
        color_d = dict(enumerate(pal, start=start))
        return color_d
    
if __name__ == '__main__':
    url = 'https://books.goalkicker.com/'
    pdf_dir = 'pdfs/'
    txt_dir = 'txt/'
    png_dir = 'figures/'

    ba = BooksAnalysis(url, pdf_dir, txt_dir, png_dir)
##    ba.get_urls()
##    ba.get_pdfs()
##    ba.to_text()
    ba.word_freq()