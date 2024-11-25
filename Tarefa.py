import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


pd.set_option('display.width', None)

df = pd.read_csv('ecommerce_preparados.csv')

print(df.head(5))
print(df.info())

print('Quantidade de Danos Nulos: \n', df.isnull().sum())
print('% de dados nulos: \n', df.isnull().mean() * 100)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

print('Estatísticas dos dados: \n', df.describe())

print('Análise de dados únicos: \n',df.nunique())

# Vendo se possui correlação entra as variáveis abaixo
df_corr = df[['Material_Cod', 'Preço', 'Desconto', 'Nota', 'Marca_Freq', 'Qtd_Vendidos_Cod', 'Temporada_Cod']].corr()

# Gráfico de Calor
plt.figure(figsize=(10,8))
sns.heatmap(df_corr, annot = True, fmt='.2f')
plt.title('Mapa de Calor da Correlação entre Variáveis')
plt.show()

df_corr2 = df[['Material_Cod', 'Preço_MinMax', 'Desconto_MinMax', 'Nota_MinMax', 'Qtd_Vendidos_Cod', 'N_Avaliações_MinMax']].corr()
plt.figure(figsize=(10,8))
sns.heatmap(df_corr, annot = True, fmt='.2f')
plt.title('Mapa de Calor da Correlação entre Variáveis')
plt.show()

# Gráfico de Dispersão
plt.hexbin(df['Preço'], df['Desconto'], gridsize = 40, cmap = 'Blues') # cmaps para utilizar: https://matplotlib.org/stable/users/explain/colors/colormaps.html
plt.colorbar(label = 'Contagem dentro do bin')
plt.xlabel('Preço')
plt.ylabel('Desconto')
plt.title('Dispersão de Preço e Desconto')
plt.show()

# Histograma
plt.hist(df['Qtd_Vendidos_Cod'])
plt.show()

# Gráfico de Barras
x = df['Nota'].value_counts().index
y = df['Nota'].value_counts().values

plt.bar(x, y, color='#90ee70')
plt.title('Divisão de Nota')
plt.xlabel('Nota')
plt.ylabel('Quantidade')

# Gráfico de Pizza
x = df['Qtd_Vendidos_Cod'].value_counts().index
y = df['Qtd_Vendidos_Cod'].value_counts().values

plt.pie(y, labels = x, autopct='%.1f%%', startangle=90)
plt.title('Distribuição de Quantidade de Vendidos')
plt.show()

# Gráfico de Densidade
plt.figure(figsize=(10,6))
sns.kdeplot(df['Preço'], fill = True, color = '#863e9c')
plt.title('Densidade de Preços')
plt.xlabel('Preço')
plt.show()

# Gráfico de Regressão
sns.regplot(x = 'Nota', y = 'Desconto', data = df, color = '#278f65', scatter_kws = {'alpha': 0.5, 'color': '#34c289'})
plt.title('Regressão de Preço e Nota')
plt.xlabel('Preço')
plt.ylabel('Nota')
plt.show()