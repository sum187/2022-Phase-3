import axios from "axios";
import { useState, useEffect } from 'react';
import * as React from 'react';
import './App.css';
import {Button,Grid,Typography,TextField} from "@mui/material";
import SearchIcon from "@mui/icons-material/Search";
import Pagination from '@mui/material/Pagination';
import Stack from '@mui/material/Stack';
import Loading from './loading';
import NoResult from './NoResultPage';
import ImageListPage from './ImageListPage';
import ImageSlider from './frontPageSlider';

function App() {
  const [movieName, setMovieName] = useState<unknown | string>();
  const [name, setName] = useState<unknown | string>();
  const [MovieInfo, setMovieInfo] = useState<unknown | any>();
  const [page, setPage] = React.useState<number>(1);
  const [pageResult, setPageResult] = React.useState<number>(0);
  const handleChange = (event: React.ChangeEvent<unknown>, value: number) => {
    setPage(value);
    setPageResult(value);
  };
  // url and key 
  const Movie_BASE_URL = "https://www.omdbapi.com";
  //const key = "938b78ef";
  const key = "8059c2e4";
  //const key = "52776e8"

  //keep track of screen size
  const [windowSize, setWindowSize] = useState(getWindowSize());
  useEffect(() => {
    function handleWindowResize() {
      setWindowSize(getWindowSize());
    }
    window.addEventListener('resize', handleWindowResize);
    return () => {
      window.removeEventListener('resize', handleWindowResize);
    };
  }, []);
  
  // keep track of selected page 
  useEffect(() => {
    if (pageResult!=0) {
      search(name,page)
    }
  }, [pageResult]);

  // loading while waiting for the result
  const [loading, setLoading] = useState(false);

  return (
    <div>
      <div className="search-field">
        <h1>Movie Search</h1 >
        <div style={{ display: "flex", justifyContent: "center" }}>
          <TextField
            id="search-bar"
            className="text"
            value={movieName}
            onChange={(prop) => {
              setMovieName(prop.target.value);
            }}
            label="Enter the moive name..."
            InputLabelProps={{style: {fontSize: fontSize()}}}
            inputProps={{style: {fontSize: fontSize()}, sx: { height: { xs: 15, sm: 15,md: 25, lg: 30, xl: 40 }}}}
            sx={{
              width: { xs: 140, sm: 140,md: 210, lg: 250, xl: 300 }
              }
            }
            variant="outlined"
            placeholder="Search..."
            size="small"
          />
          <Button
            onClick={() => {
              // movie name not entered, do nothing
              if (movieName===''){
                <div></div>
              // new movie name is searched
              } else if (movieName!==name) {
                search(movieName,page),
                setPage(1),
                setName(movieName)
              // same movie name is searched but not at page 1
              } else if (movieName===name&&page!==1){
                setPage(1)
                search(movieName,1)
              }
            }}
          >
            <SearchIcon style={{ fill: "blue" }} />
            Search
          </Button>
        </div>
      </div>
      {MovieInfo === undefined? (
        <ImageSlider 
          width={widthRatio('width')} 
          height={widthRatio('height')} 
        />
      ) : loading===true? (
        <Loading />
      ) : MovieInfo.Response === "False" ? (
        <NoResult 
          name={name} 
          width={widthRatio('width')}
        />
      ) : MovieInfo.Response === "True"? (
        <><ImageListPage
            MovieInfo={MovieInfo}
            col={getColumn(windowSize.innerWidth)}
            />
            <Grid
              container
              direction="row"
              spacing={0}
              sx={{justifyContent: "center",}}>
              <Stack spacing={2}>
                <Typography>Page: {page}</Typography>
                <Pagination count={Math.ceil(parseInt(MovieInfo.totalResults) / 30)}
                  variant="outlined"
                  shape="rounded"
                  page={page}
                  onChange={handleChange} 
                  />
              </Stack>
            </Grid></>         
      ):(
        <div></div>
      )}   
    </div>
  );

  
  async function search(name:any,page:any) {
    // loading images
    setLoading(true)
    console.log('loading')

    // get response for first time
    const res = await axios.
    get(Movie_BASE_URL + "/?s=" + name + "&apikey=" + key + "&page=" +page); 
    
    // check the response  
    if (res.data.Response!=='True'){
      console.log('there was no search');
      setMovieInfo(res.data);
    } else {
    var value=0 

    // making sure page number is not exceed
    const maxPage=Math.ceil(res.data.totalResults/10)
    var endPage=page*3
    if (endPage>maxPage){
      endPage=maxPage
    }

      // loop to call multiple responses (each response has maximum of 10 search results)
      // extracting 30 results, if less than 30, all results are extracted
      for (let i = page*3-2; i < endPage+1; i++) {

        // get response
        var response = await axios.
          get(Movie_BASE_URL + "/?s=" + name + "&apikey=" + key + "&page=" +i.toString());

        // loop to combine responses
        for (let j=0; j<response.data.Search.length;j++){
          res.data.Search[value]=response.data.Search[j]
          value+=1;
        }
      }
      setMovieInfo(res.data);
    }
    // loading finished
    console.log('finish')
    setLoading(false)
 }  

  function getWindowSize() {
    const {innerWidth, innerHeight} = window;
    return {innerWidth, innerHeight};
  }
  
  // get number of image in the column depending on screen size
  function getColumn(width:any) {
    if (width<320) {
      return 1;
    } else if (width>=320&&width<=480) {
      return 2;
    } else if (width>480&&width<=768) {
      return 3;
    } else if (width>=768&&width<=1024) {
      return 5;
    } else if (width>=1024) {
      return 6;
    }}

  // get width depeding on screen width
  function widthRatio(key:string) {
    const width:number=windowSize.innerWidth
    var value:number=0
    if (width<320) {
      value= 0.9*windowSize.innerWidth;
    } else if (width>=320&&width<=480) {
      value= 0.8*windowSize.innerWidth;
    } else if (width>480&&width<=768) {
      value= 0.7*windowSize.innerWidth;
    } else if (width>=768&&width<=1024) {
      value= 0.6*windowSize.innerWidth;
    } else if (width>=1024) {
      value= 0.5*windowSize.innerWidth;
    }
    if (key=='width') {
      return value;
    } else if (key=='height') {
      return value*1.4;
    }
  }  

    function fontSize() {
      const col=getColumn(window.innerWidth)
      if (col==1) {
        return 10
      } else if (col==2) {
        return 10
      } else if (col==3) {
        return 17
      } else if (col==5) {
        return 20
      } else if (col==6) {
        return 25
      }}  
}

export default App;
