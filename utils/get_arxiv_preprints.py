import arxiv
import pandas as pd

def df_from_query(query, num_of_results=10, sort_by_relevance=True, save_path=None):
    """
    extract information from preprints using arxiv API
    Args: 
        query (str): the query to search for
        num_of_results (int): the number of results that you want
        save_path (str): a path where you want to save the df, or None in order not to save the df
    Returns:
        df (pandas DataFrame): a dataframe with information for the papers
        
    Example:
        df = df_from_query(query='forcasting', num_of_results=50, save_path='arxiv_preprints.csv')
    """
    search = arxiv.Search(
        query = query,
        max_results = num_of_results,
        sort_by = arxiv.SortCriterion.Relevance if sort_by_relevance else arxiv.SortCriterion.SubmittedDate
    )

    df = pd.DataFrame(columns=['entry_id', 'title', 'abstract', 'authors', 'categories', 'pub_date', 'pdf_url'])
    for i, result in enumerate(search.results()):
        #result.download_pdf(dirpath=dir_path)
        title = result.title
        authors = result.authors
        abstract = result.summary
        categories = result.categories
        pdf_url = result.pdf_url
        # convert lists to strings
        authors_ = ", ".join(str(x) for x in authors)
        categories_ = " ".join(str(x) for x in categories)
        
        df.loc[i, 'entry_id'] = result.entry_id
        df.loc[i, 'pub_date'] = result.published.strftime("%Y-%m-%d")
        df.loc[i, 'title'] = title
        df.loc[i, 'abstract'] = abstract.replace("\n", " ")
        df.loc[i, 'authors'] = authors_
        df.loc[i, 'categories'] = categories_
        df.loc[i, 'pdf_url'] = pdf_url
        # import pdb; pdb.set_trace()
        
    if save_path: df.to_csv(save_path, index=False)

    return df 